from kedro.pipeline import Pipeline, node, pipeline
from .train import create_dataloaders, run_training_loop, run_final_evaluation

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_dataloaders,
            inputs="params:learning_settings",
            outputs="data_loaders",
            name="create_data_node"
        ),
        node(
            func=run_training_loop,
            inputs=["data_loaders", "params:learning_settings", "params:vit_model"],
            outputs="trained_model_vit",
            name="training_node"
        ),
        node(
            func=run_final_evaluation,
            inputs=["trained_model_vit", "data_loaders", "params:learning_settings"],
            outputs="final_metrics_vit",
            name="evaluation_node"
        )
    ])