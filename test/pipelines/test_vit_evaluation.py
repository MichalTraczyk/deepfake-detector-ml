import pytest
import torch
from unittest.mock import MagicMock, patch
from deepfake_detector.pipelines.vit_training.train import run_final_evaluation

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.parameters.return_value = [torch.randn(1, requires_grad=True)]
    return model


@pytest.fixture
def mock_loaders():
    return {
        "test": MagicMock()
    }


@pytest.fixture
def mock_params():
    return {
        "checkpoint_path_vit": "dummy/path/to/checkpoint.pt",
        "checkpoint_path": "dummy/path/to/checkpoint.pt"
    }


# --- Testy ---

@patch('deepfake_detector.pipelines.vit_training.train.evaluate_model_metrics')
@patch('deepfake_detector.pipelines.vit_training.train.load_checkpoint')
def test_run_final_evaluation_flow(mock_load_checkpoint, mock_evaluate_metrics, mock_model, mock_loaders, mock_params):
    mock_load_checkpoint.return_value = (mock_model, MagicMock(), 20)

    expected_metrics = {"accuracy": 0.95, "loss": 0.15}
    mock_evaluate_metrics.return_value = expected_metrics

    result = run_final_evaluation(mock_model, mock_loaders, mock_params)

    mock_load_checkpoint.assert_called_once()

    call_args = mock_load_checkpoint.call_args

    assert call_args[0][2] == "dummy/path/to/checkpoint.pt"

    mock_evaluate_metrics.assert_called_once()

    assert result == expected_metrics
    assert result['accuracy'] == 0.95


@patch('deepfake_detector.pipelines.vit_training.train.load_checkpoint')
def test_evaluation_handles_missing_checkpoint_param(mock_load_checkpoint, mock_model, mock_loaders):
    bad_params = {"wrong_key": "path"}

    with pytest.raises(KeyError):
        run_final_evaluation(mock_model, mock_loaders, bad_params)