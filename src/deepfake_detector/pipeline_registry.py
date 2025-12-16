"""Project pipelines."""
from kedro.pipeline import Pipeline
from .pipelines import vit_training, cnn_training, cnn_test, vit_test
def register_pipelines() -> dict[str, Pipeline]:
    training_pipeline_vit = vit_training.create_pipeline()
    training_pipeline_cnn = cnn_training.create_pipeline()
    test_pipeline_cnn = cnn_test.create_pipeline()
    test_pipeline_vit = cnn_test.create_pipeline()
    return {
        "training_vit": training_pipeline_vit,
        "training_cnn": training_pipeline_cnn,
        "test_cnn": test_pipeline_cnn,
        "test_vit": test_pipeline_vit
    }
