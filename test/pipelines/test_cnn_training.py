import pytest
import torch
from unittest.mock import patch, MagicMock
from deepfake_detector.pipelines.cnn_training.train import create_dataloaders, train


@pytest.fixture
def mock_cnn_params():
    return {
        "image_resolution": 224,
        "batch_size": 4,
        "data_dir": "/dummy/path",
        "learning_rate": 0.001,
        "num_epochs": 1,
        "checkpoint_path_cnn": "dummy_checkpoint.pt"
    }


@patch('deepfake_detector.pipelines.cnn_training.train.ImageFolder')
def test_create_cnn_dataloaders(mock_image_folder, mock_cnn_params):
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.__len__.return_value = 10
    mock_image_folder.return_value = mock_dataset_instance

    loaders = create_dataloaders(mock_cnn_params)

    assert isinstance(loaders, dict)
    assert "train" in loaders
    assert "val" in loaders

    assert loaders["val"].batch_size == 4

@patch('deepfake_detector.pipelines.cnn_training.train.save_checkpoint')
@patch('deepfake_detector.pipelines.cnn_training.train.train_one_epoch')
@patch('deepfake_detector.pipelines.cnn_training.train.evaluate_train_accuracy')
@patch('deepfake_detector.pipelines.cnn_training.train.load_checkpoint')
@patch('deepfake_detector.pipelines.cnn_training.train.MultiInputModel')
def test_cnn_training_flow(mock_model_cls, mock_load_checkpoint, mock_eval, mock_train_epoch, mock_save_checkpoint, mock_cnn_params):
    mock_model_instance = MagicMock()

    dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
    mock_model_instance.parameters.return_value = [dummy_param]
    mock_model_instance.to.return_value = mock_model_instance

    mock_model_cls.return_value = mock_model_instance

    mock_load_checkpoint.return_value = (mock_model_instance, MagicMock(), 0)

    mock_train_epoch.return_value = (0.5, 0.8)
    mock_eval.return_value = (0.4, 0.85)

    mock_loaders = {
        "train": MagicMock(),
        "val": MagicMock()
    }

    result_model = train(mock_loaders, mock_cnn_params)

    assert result_model == mock_model_instance
    mock_train_epoch.assert_called_once()
    mock_eval.assert_called_once()
    mock_save_checkpoint.assert_called()