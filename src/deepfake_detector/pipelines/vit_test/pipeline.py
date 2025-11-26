from kedro.pipeline import Pipeline, node, pipeline
from .test import create_vit_gradcam_visualization

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_vit_gradcam_visualization,
            inputs=[
                "trained_model_vit",
                "data_loaders",
                "params:learning_settings",
                "params:vit_model"
            ],
            outputs="vit_gradcam_image",
            name="generate_vit_gradcam_node"
        )
    ])