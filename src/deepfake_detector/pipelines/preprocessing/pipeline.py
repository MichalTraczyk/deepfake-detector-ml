from kedro.pipeline import Pipeline, node, pipeline

from deepfake_detector.pipelines.preprocessing.preprocess import run_extraction, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_extraction,
            inputs={
                "raw_dirs": "params:preprocess.celeb_df_dirs",
                "target_base_dir": "params:preprocess.celeb_df_output",
                "res": "params:learning_settings.image_resolution",
                "mode": "params:preprocess.mode_multi"
            },
            outputs="celeb_df_intermediate",
            name="extract_celeb_df"
        ),
        node(
            func=run_extraction,
            inputs={
                "raw_dirs": "params:preprocess.forensics_dirs",
                "target_base_dir": "params:preprocess.forensics_output",
                "res": "params:learning_settings.image_resolution",
                "mode": "params:preprocess.mode_single"
            },
            outputs="forensics_intermediate",
            name="extract_forensics"
        ),
        node(
            func=split_data,
            inputs=["celeb_df_intermediate", "params:preprocess.train_ratio"],
            outputs="celeb_df_final_status",
            name="split_celeb_df"
        )
    ])