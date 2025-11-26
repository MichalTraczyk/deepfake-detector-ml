import torch
import pytest
from deepfake_detector.modules.cnn.model_cnn import MultiInputModel


def test_cnn_initialization():
    model = MultiInputModel()
    assert model is not None


def test_cnn_forward_tensor_shapes():
    model = MultiInputModel()

    rgb_dummy = torch.randn(2, 3, 256, 256)
    fft_dummy = torch.randn(2, 3, 256, 256)

    input_dict = {
        'rgb_input': rgb_dummy,
        'fft_input': fft_dummy
    }

    output = model(input_dict)

    assert output.shape == (2, 1)


def test_cnn_forward_missing_key():
    model = MultiInputModel()
    rgb_dummy = torch.randn(1, 3, 256, 256)

    bad_dict = {'rgb_input': rgb_dummy}

    with pytest.raises(KeyError):
        model(bad_dict)