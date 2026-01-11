from kedro.pipeline import Pipeline, node, pipeline
from .test import load_vit_model_node, create_test_dataloader_node, create_vit_gradcam_plot_node

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_test_dataloader_node,
            inputs="params:learning_settings",
            outputs="vit_test_loader",
            name="create_vit_dataloader_node",
        ),
        node(
            func=load_vit_model_node,
            inputs=["params:vit_model", "params:paths", "params:learning_settings"],
            outputs="loaded_vit_model",
            name="load_vit_model_node",
        ),
        node(
            func=create_vit_gradcam_plot_node,
            inputs=["loaded_vit_model", "vit_test_loader"],
            outputs="vit_gradcam_plot",
            name="create_vit_gradcam_visualization_node",
        ),
    ])