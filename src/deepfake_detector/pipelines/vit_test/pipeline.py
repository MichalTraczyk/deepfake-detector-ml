from kedro.pipeline import Pipeline, node, pipeline
from .test import load_vit_model_node, create_test_dataloader_node, create_vit_gradcam_plot_node, run_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    """
    Funkcja do tworzenia pipeline'u do testów dla modelu ViT.
    """
    return pipeline([
        node(
            func=create_test_dataloader_node,
            inputs=["params:learning_settings", "params:preprocess"],
            outputs=["test_dataloader_celeb_df", "test_dataloader_celeb_ff"],
            name="create_vit_dataloader_node",
        ),
        node(
            func=load_vit_model_node,
            inputs=["params:vit_model", "params:paths", "params:learning_settings"],
            outputs="loaded_vit_model",
            name="load_vit_model_node",
        ),
        node(
            func=run_evaluation,
            inputs=["loaded_vit_model", "test_dataloader_celeb_df"],
            outputs="final_metrics_vit_celeb"
        ),
        node(
            func=run_evaluation,
            inputs=["loaded_vit_model", "test_dataloader_celeb_ff"],
            outputs="final_metrics_vit_ff"
        ),
        node(
            func=create_vit_gradcam_plot_node,
            inputs=["loaded_vit_model", "test_dataloader_celeb_df"],
            outputs="vit_gradcam_plot",
            name="create_vit_gradcam_visualization_node",
        ),
    ])