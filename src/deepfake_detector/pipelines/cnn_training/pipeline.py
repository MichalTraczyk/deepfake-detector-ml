from kedro.pipeline import Pipeline, node, pipeline
from .train import create_dataloaders, train, run_final_evaluation

def create_pipeline(**kwargs) -> Pipeline:
    """
       Funkcja do tworzenia pipeline'u do treningu modelu CNN.
    """
    return pipeline([
        node(
            func=create_dataloaders,
            inputs="params:learning_settings",
            outputs="data_loaders",
            name="create_data_node"
        ),
        node(
            func=train,
            inputs=["data_loaders", "params:learning_settings"],
            outputs="trained_model_cnn",
            name="train_cnn"
        ),
        node(
            func=run_final_evaluation,
            inputs=["trained_model_cnn", "data_loaders", "params:learning_settings"],
            outputs="final_evaluation_cnn",
            name="final_evaluation_cnn"
        )
    ])