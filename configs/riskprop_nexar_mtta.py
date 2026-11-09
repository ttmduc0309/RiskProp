"""Nexar full-timeline inference for timestamp-based test mTTA."""

_base_ = ["riskprop_nexar.py"]

algorithm_keys = (
    "dataset", "frame_dir", "filename_tmpl", "img_shape", "sample_idx", "video_id",
    "type", "start_index", "total_frames", "target", "abnormal_start_frame",
    "accident_frame", "frame_inds", "clip_len", "num_clips", "frame_interval",
    "fps", "is_val", "is_test", "text", "onset_time_sec", "event_time_sec",
    "test_partition", "frame_timestamps_sec",
)

mtta_test_pipeline = [
    dict(type="DecordInit", io_backend="disk"),
    dict(type="SampleFullVideoCausalFrames", clip_len=5),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flow", modality="rgb"),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]

test_dataloader = dict(
    # Nexar clip durations vary, so full-timeline tensors are not batch-stackable.
    batch_size=1,
    dataset=dict(
        pipeline_video=mtta_test_pipeline,
        pipeline_frame=mtta_test_pipeline,
        test_only=True,
    ),
)

test_evaluator = dict(
    type="AnticipationMetric",
    fpr_max=0.1,
    vis_list={},
    prediction_file="work_dirs/riskprop_nexar_mtta/test_predictions.csv",
    temporal_prediction_file="work_dirs/riskprop_nexar_mtta/test_temporal_predictions.jsonl",
    compute_test_mtta=True,
)

work_dir = "work_dirs/riskprop_nexar_mtta"
