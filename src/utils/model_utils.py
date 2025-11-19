import torch
import torch.nn.functional as F
import torchvision.models as models
from src.models.vit.model_vit import ModelViT


def load_pretrained_weights(model: ModelViT) -> ModelViT:

    try:
        weights = models.VisionTransformer_Weights.IMAGENET1K_V1
        pretrained_dict = weights.get_state_dict()
    except AttributeError:
        temp_model = models.vit_b_16(pretrained=True)
        pretrained_dict = temp_model.state_dict()

    key_mapping = {
        'class_token': 'cls_token',
        'conv_proj.weight': 'patch_embed.projection.weight',
        'conv_proj.bias': 'patch_embed.projection.bias',
        'encoder.pos_embedding': 'pos_embedding'
    }

    for old_key in list(pretrained_dict.keys()):
        if old_key in key_mapping:
            new_key = key_mapping[old_key]
            pretrained_dict[new_key] = pretrained_dict.pop(old_key)

        elif 'encoder.layers.encoder_layer_' in old_key:
            pass

    keys_to_remove = ['heads.head.weight', 'heads.head.bias', 'head.weight', 'head.bias']
    for k in keys_to_remove:
        if k in pretrained_dict:
            del pretrained_dict[k]
    if 'pos_embedding' in pretrained_dict:
        if model.pos_embedding.shape != pretrained_dict['pos_embedding'].shape:

            pos_embed_pretrained = pretrained_dict['pos_embedding']

            cls_token = pos_embed_pretrained[:, 0:1]
            patch_embed = pos_embed_pretrained[:, 1:]

            orig_size = int(patch_embed.shape[1] ** 0.5)
            new_size = int((model.pos_embedding.shape[1] - 1) ** 0.5)

            embed_dim = model.embed_dim

            patch_embed = patch_embed.transpose(1, 2).reshape(1, embed_dim, orig_size, orig_size)
            patch_embed = F.interpolate(patch_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            patch_embed = patch_embed.flatten(2).transpose(1, 2)

            pretrained_dict['pos_embedding'] = torch.cat((cls_token, patch_embed), dim=1)
    else:
        print("Brak pos_embedding")

    msg = model.load_state_dict(pretrained_dict, strict=False)

    return model