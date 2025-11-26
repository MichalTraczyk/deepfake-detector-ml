import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class ModelViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, depth, num_heads, mlp_dim, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.heads = nn.Sequential(nn.Linear(embed_dim, 1))

    def forward(self, x):
        if isinstance(x, dict):
            if 'rgb_input' in x:
                x = x['rgb_input']
            else:
                raise KeyError(f"Brak odpowiedniego klucza")

        B = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.pos_dropout(x)
        x = self.transformer_encoder(x)
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)

        logits = self.heads(cls_output)
        return logits