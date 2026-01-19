import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


def load_pretrained_weights(model):
    """
    Wczytywanie przetrenowanych wag z modeli ViT-B/16 do modelu Vision Transformer

    Funkcja ta tłumaczy strukturę wag z przetrenowanego modelu z biblioteki 'torchvision'
    na strukturę stworzoną w klasie 'ModelVit'

    Najpierw pobierane są wagi wytrenowane na ImageNet-1k, następnie tłumaczy nazwy warstw,
    po tym dochodzi do interpolacji pozycji, jeśli dany model przyjmuje obrazy o innej
    rozdzielczości niż przetrenowany model oraz na ostatnim etapie ignoruje wagi ostatniej
    warstwy klasyfikacyjnej ze względu, że ImageNet ma 1000 klas, a stworzony model ma tylko 2.

    Args:
        model (nn.Module): Instancja klasy 'ModelVit'

    Returns:
        nn.Module: Model z przypisanymi wagami

    """

    # Pobieranie wag z torchvision
    try:
        weights = models.VisionTransformer_Weights.IMAGENET1K_V1
        pretrained_dict = weights.get_state_dict()
    except AttributeError:
        # Pobieranie dla starszych wersji torchvision ze względu na trenowanie na różnych komputerach
        temp_model = models.vit_b_16(pretrained=True)
        pretrained_dict = temp_model.state_dict()

    new_state_dict = {}

    # Tłumaczenie nazw warstw dla stworzonego modelu
    for key, value in pretrained_dict.items():
        new_key = key

        if key == 'class_token':
            new_key = 'embeddings_block.cls_token'

        elif key == 'encoder.pos_embedding':
            new_key = 'embeddings_block.position_embeddings'

        elif key.startswith('conv_proj'):
            new_key = key.replace('conv_proj', 'embeddings_block.patcher.0')

        elif key.startswith('encoder.layers.encoder_layer_'):
            parts = key.split('.')
            layer_idx = parts[2].split('_')[-1]

            new_prefix = f"encoder_blocks.layers.{layer_idx}"

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

    pos_key_model = 'embeddings_block.position_embeddings'

    if pos_key_model in new_state_dict:
        pos_embed_pretrained = new_state_dict[pos_key_model]

        pos_embed_model = model.embeddings_block.position_embeddings

        if pos_embed_model.shape != pos_embed_pretrained.shape:

            cls_token = pos_embed_pretrained[:, 0:1, :]
            patch_tokens = pos_embed_pretrained[:, 1:, :]

            embed_dim = pos_embed_model.shape[2]

            num_tokens_old = patch_tokens.shape[1]
            gs_old = int(math.sqrt(num_tokens_old))
            num_tokens_new = pos_embed_model.shape[1] - 1
            gs_new = int(math.sqrt(num_tokens_new))

            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(1, embed_dim, gs_old, gs_old)

            patch_tokens = F.interpolate(
                patch_tokens,
                size=(gs_new, gs_new),
                mode='bicubic',
                align_corners=False
            )

            patch_tokens = patch_tokens.flatten(2).permute(0, 2, 1)

            new_pos_embed = torch.cat((cls_token, patch_tokens), dim=1)

            new_state_dict[pos_key_model] = new_pos_embed

    msg = model.load_state_dict(new_state_dict, strict=False)

    missing_encoder = [k for k in msg.missing_keys if 'encoder_blocks' in k]
    if len(missing_encoder) > 0:
        print(f"Brakuje {len(missing_encoder)}")
        print(f"Przykłady: {missing_encoder[:3]}")
    else:
        print("Wagi transformera załadowane poprawnie")

    return model