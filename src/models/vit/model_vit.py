import torch
import torch.nn as nn
from config import ViTConfig


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class ModelViT(nn.Module):

    def __init__(self, config: ViTConfig):

        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size, config.in_channels, config.embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        self.pos_dropout = nn.Dropout(p=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.depth
        )
        self.norm = nn.LayerNorm(config.embed_dim)
        self.mlp_head = nn.Linear(config.embed_dim, config.num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        x = self.transformer_encoder(x)

        cls_output = x[:, 0]

        cls_output = self.norm(cls_output)
        logits = self.mlp_head(cls_output)

        return logits