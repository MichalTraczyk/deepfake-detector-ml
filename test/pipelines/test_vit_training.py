import pytest
from unittest.mock import patch, MagicMock
from deepfake_detector.pipelines.vit_training.train import create_dataloaders


@pytest.fixture
def mock_params():
    return {
        "image_resolution": 224,
        "batch_size": 4,
        "data_dir": "/dummy/path"
    }


@patch('deepfake_detector.pipelines.vit_training.train.ImageFolder')
def test_create_dataloaders(mock_image_folder, mock_params):
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.__len__.return_value = 10
    mock_image_folder.return_value = mock_dataset_instance

    loaders = create_dataloaders(mock_params)

    assert isinstance(loaders, dict)
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    assert loaders["train"].batch_sampler.batch_size == 4
    assert loaders["val"].batch_size == 4