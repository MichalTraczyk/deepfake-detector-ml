import pytest
import torch
from unittest.mock import MagicMock, patch
from deepfake_detector.pipelines.cnn_training.train import run_final_evaluation


@pytest.fixture
def mock_cnn_model():
    model = MagicMock()
    dummy_param = torch.nn.Parameter(torch.tensor([1.0]))
    model.parameters.return_value = [dummy_param]
    return model


@pytest.fixture
def mock_loaders():
    return {
        "test": MagicMock()
    }


@pytest.fixture
def mock_params():
    return {
        "checkpoint_path_cnn": "dummy/path/to/cnn_checkpoint.pt"
    }


# --- Testy ---

@patch('deepfake_detector.pipelines.cnn_training.train.evaluate_model_metrics')
@patch('deepfake_detector.pipelines.cnn_training.train.load_checkpoint')
def test_cnn_final_evaluation_flow(mock_load_checkpoint, mock_evaluate_metrics, mock_cnn_model, mock_loaders,
                                   mock_params):
    mock_load_checkpoint.return_value = (mock_cnn_model, MagicMock(), 10)

    expected_metrics = {"accuracy": 0.88, "f1": 0.85}
    mock_evaluate_metrics.return_value = expected_metrics

    result = run_final_evaluation(mock_cnn_model, mock_loaders, mock_params)

    mock_load_checkpoint.assert_called_once()

    assert mock_load_checkpoint.call_args[0][2] == "dummy/path/to/cnn_checkpoint.pt"

    mock_evaluate_metrics.assert_called_once()

    assert result == expected_metrics


def test_cnn_evaluation_missing_config(mock_cnn_model, mock_loaders):
    bad_params = {"wrong_key": "value"}

    with pytest.raises(KeyError):
        run_final_evaluation(mock_cnn_model, mock_loaders, bad_params)