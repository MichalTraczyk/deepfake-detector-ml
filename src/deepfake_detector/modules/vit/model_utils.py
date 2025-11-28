import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .model_vit import ModelViT


def load_pretrained_weights(model: ModelViT) -> ModelViT:
    print("Wczytywanie wag pretrained (ViT) z torchvision...")

    try:
        weights = models.VisionTransformer_Weights.IMAGENET1K_V1
        pretrained_dict = weights.get_state_dict()
    except AttributeError:
        print("Starsza wersja torchvision: pobieranie wag z modelu...")
        temp_model = models.vit_b_16(pretrained=True)
        pretrained_dict = temp_model.state_dict()

    new_state_dict = {}

    for key, value in pretrained_dict.items():
        new_key = key

        if key == 'class_token':
            new_key = 'cls_token'
        elif key.startswith('conv_proj'):
            new_key = key.replace('conv_proj', 'patch_embed.projection')
        elif key == 'encoder.pos_embedding':
            new_key = 'pos_embedding'

        elif key.startswith('encoder.layers.encoder_layer_'):
            parts = key.split('.')
            layer_idx = parts[2].split('_')[-1]

            new_prefix = f"transformer_encoder.layers.{layer_idx}"

            suffix = ".".join(parts[3:])

            suffix = suffix.replace('ln_1', 'norm1')
            suffix = suffix.replace('ln_2', 'norm2')

            suffix = suffix.replace('self_attention', 'self_attn')

            if 'mlp.linear_1' in suffix:
                suffix = suffix.replace('mlp.linear_1', 'linear1')
            elif 'mlp.0' in suffix:
                suffix = suffix.replace('mlp.0', 'linear1')

            if 'mlp.linear_2' in suffix:
                suffix = suffix.replace('mlp.linear_2', 'linear2')
            elif 'mlp.3' in suffix:
                suffix = suffix.replace('mlp.3', 'linear2')

            new_key = f"{new_prefix}.{suffix}"

        if 'heads' in key or 'head' in key:
            continue

        new_state_dict[new_key] = value

    if 'pos_embedding' in new_state_dict:
        if model.pos_embedding.shape != new_state_dict['pos_embedding'].shape:
            print(f"Interpolacja osadzeń: {new_state_dict['pos_embedding'].shape} -> {model.pos_embedding.shape}")
            pos_embed_pretrained = new_state_dict['pos_embedding']
            cls_token = pos_embed_pretrained[:, 0:1]
            patch_embed = pos_embed_pretrained[:, 1:]
            orig_size = int(patch_embed.shape[1] ** 0.5)
            new_size = int((model.pos_embedding.shape[1] - 1) ** 0.5)
            embed_dim = model.embed_dim
            patch_embed = patch_embed.transpose(1, 2).reshape(1, embed_dim, orig_size, orig_size)
            patch_embed = F.interpolate(patch_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            patch_embed = patch_embed.flatten(2).transpose(1, 2)
            new_state_dict['pos_embedding'] = torch.cat((cls_token, patch_embed), dim=1)

    msg = model.load_state_dict(new_state_dict, strict=False)

    missing_encoder = [k for k in msg.missing_keys if 'transformer_encoder' in k]
    if len(missing_encoder) > 0:
        print(f"Nadal brakuje {len(missing_encoder)} kluczy encodera!")
        print(f"Przykłady braków: {missing_encoder[:3]}")
    else:
        print("Wszystkie warstwy Transformera załadowane poprawnie!")

    return model