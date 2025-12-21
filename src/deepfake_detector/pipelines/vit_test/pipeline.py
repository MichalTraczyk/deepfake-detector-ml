from kedro.pipeline import Pipeline, node, pipeline
from .test import create_vit_gradcam_visualization, get_test_model, get_test_dataloader, run_evaluation


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_test_dataloader,
            inputs=["params:learning_settings"],
            outputs="test_dataloader",
            name="loaders_node"
        ),
        node(
            func=get_test_model,
            inputs=["params:learning_settings","params:vit_model","params:paths"],
            outputs="test_model",
            name="model_node"
        ),
        node(
            func=run_evaluation,
            inputs=["test_model", "test_dataloader"],
            outputs="final_metrics_vit"
        ),
        # node(
        #     func=create_vit_gradcam_visualization,
        #     inputs=[
        #         "test_model",
        #         "test_dataloader",
        #         "params:learning_settings",
        #         "params:vit_model"
        #     ],
        #     outputs="vit_gradcam_image",
        #     name="generate_vit_gradcam_node"
        # )
    ])