"""Project pipelines."""
from kedro.pipeline import Pipeline
from .pipelines import vit_training, cnn_training


def register_pipelines() -> dict[str, Pipeline]:
    training_pipeline_vit = vit_training.create_pipeline()
    training_pipeline_cnn = cnn_training.create_pipeline()

    return {
        "training_vit": training_pipeline_vit,
        "training_cnn": training_pipeline_cnn
    }
