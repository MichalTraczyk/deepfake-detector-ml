from kedro.pipeline import Pipeline, node, pipeline
from .test import create_cnn_gradcam_visualization

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_cnn_gradcam_visualization,
            inputs="params:learning_settings",
            outputs="cnn_gradcam_plot",
            name="cnn_gradcam_node"
        )
    ])