"""One-epoch wiring check on a small, balanced Nexar subset."""

_base_ = ["_base_/schedules/sgd_50e.py", "_base_/default_runtime.py"]

custom_imports = dict(imports="taa", allow_failed_imports=False)

data_root = "data/nexar-collision-prediction"
nexar = dict(data_root=data_root, ann_file="annotations.csv", filename_tmpl=None, start_index=0)
clip_len = 5
num_clips = 30
algorithm_keys = (
    "dataset", "frame_dir", "filename_tmpl", "img_shape", "sample_idx", "video_id",
    "type", "start_index", "total_frames", "target", "abnormal_start_frame",
    "accident_frame", "frame_inds", "clip_len", "num_clips", "frame_interval",
    "fps", "is_val", "is_test", "text",
)
file_client_args = dict(io_backend="disk")

train_pipeline_video = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFramesBeforeAccident", clip_len=clip_len, num_clips=num_clips),
    dict(type="DecordDecode"),
    dict(type="RandomResizedCrop", area_range=(0.8, 1.0), aspect_ratio_range=(4 / 3, 16 / 9)),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flip", flip_ratio=0.5),
    dict(type="Flow", modality="rgb"),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]
val_pipeline_video = [
    dict(type="DecordInit", **file_client_args),
    dict(type="SampleFramesBeforeAccident", clip_len=clip_len, num_clips=num_clips, test_mode=True),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(224, 224), keep_ratio=False),
    dict(type="Flow", modality="rgb"),
    dict(type="FormatShape", input_format="NCTHW"),
    dict(type="PackActionInputs", meta_keys=(), algorithm_keys=algorithm_keys),
]

train_ids = ["train:00822", "train:00208", "train:01924", "train:01904"]
val_ids = [
    "train:00128", "train:00457", "train:00900", "train:00098", "train:00100",
    "train:00765", "train:01022", "train:00629", "train:00292", "train:00600",
    "train:00335", "train:00029", "train:00960", "train:00702", "train:00819",
    "train:00459", "train:00370", "train:00477", "train:00587", "train:00963",
    "train:01429", "train:01486", "train:01755", "train:01438", "train:01653",
    "train:01405", "train:01276", "train:02067", "train:02026", "train:01717",
    "train:01728", "train:02049", "train:01138", "train:01292", "train:02139",
    "train:01600", "train:01988", "train:01219", "train:02121", "train:01444",
    "test:00002", "test:01044",
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True, round_up=True),
    dataset=dict(
        type="MultiDataset", cap=None, dada=None, d2city=None, nexar=nexar,
        pipeline_video=train_pipeline_video, pipeline_frame=train_pipeline_video,
        modality="rgb", test_mode=False, train_with_val=False, video_ids=train_ids,
    ),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="MultiDataset", cap=None, dada=None, d2city=None, nexar=nexar,
        pipeline_video=val_pipeline_video, pipeline_frame=val_pipeline_video,
        modality="rgb", test_mode=True, val_train=False, video_ids=val_ids,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="AnticipationMetric", fpr_max=0.1, vis_list={},
    prediction_file="work_dirs/riskprop_nexar_smoke/test_predictions.csv",
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_begin=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=2, save_best="mAUC@", rule="greater")
)
custom_hooks = [dict(type="EpochHook"), dict(type="AnticipationMetricHook")]
work_dir = "work_dirs/riskprop_nexar_smoke"

model = dict(
    type="Recognizer3D",
    backbone=dict(
        type="ResNet3dSlowOnly", depth=50,
        pretrained="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        lateral=False, conv1_kernel=(1, 7, 7), conv1_stride_t=1, pool1_stride_t=1,
        inflate=(0, 0, 1, 1), norm_eval=False,
    ),
    cls_head=dict(
        type="AnticipationHead", pos_weight=1, clip_len=clip_len, num_clips=num_clips,
        two_stream=False, with_rnn=False, with_decoder=False, label_with="constraint",
    ),
    data_preprocessor=dict(
        type="ActionDataPreprocessor", mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375], format_shape="NCTHW",
    ),
    train_cfg=None, test_cfg=None,
)
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb_20230612-12ce977c.pth"
