import torch
import pytest
from deepfake_detector.modules.vit.model_vit import ModelViT


@pytest.fixture
def vit_config():
    return {
        "image_size": 64,
        "patch_size": 8,
        "in_channels": 3,
        "embed_dim": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_dim": 64,
        "dropout": 0.1
    }


def test_model_initialization(vit_config):
    model = ModelViT(**vit_config)
    assert model is not None
    assert model.embed_dim == 32


def test_model_forward_tensor(vit_config):
    model = ModelViT(**vit_config)
    dummy_input = torch.randn(2, 3, 64, 64)

    output = model(dummy_input)

    assert output.shape == (2, 1)


def test_model_forward_dict(vit_config):
    model = ModelViT(**vit_config)
    dummy_input = torch.randn(2, 3, 64, 64)

    input_dict = {'rgb_input': dummy_input}

    output = model(input_dict)
    assert output.shape == (2, 1)


def test_model_missing_key(vit_config):
    model = ModelViT(**vit_config)
    dummy_input = torch.randn(1, 3, 64, 64)
    bad_dict = {'wrong_key': dummy_input}

    with pytest.raises(KeyError):
        model(bad_dict)