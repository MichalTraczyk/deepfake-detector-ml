from kedro.pipeline import Pipeline, node, pipeline
from .test import create_cnn_gradcam_visualization, get_test_dataloader, get_test_model, run_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
        func=get_test_dataloader,
        inputs="params:learning_settings",
        outputs="test_dataloader",
        name="loaders_node"
    ),
        node(
            func=get_test_model,
            inputs="params:paths",
            outputs="test_model",
            name="cnn_model_node"
        ),
        node(
            func=run_evaluation,
            inputs=["test_model", "test_dataloader"],
            outputs="final_metrics_cnn"
        ),
        node(
            func=create_cnn_gradcam_visualization,
            inputs=["test_dataloader", "test_model"],
            outputs="cnn_gradcam_plot",
            name="cnn_gradcam_plot"
        )
    ])
