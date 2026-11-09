"""Full Nexar training/validation setup; not a paper-exact reproduction recipe."""

_base_ = ["riskprop_nexar_smoke.py"]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(video_ids=None),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(video_ids=None, include_test=False),
)
test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(video_ids=None, include_test=True),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_begin=1, val_interval=1)
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=50, save_best="mAUC_full@", rule="greater")
)
val_evaluator = dict(prediction_file=None)
test_evaluator = dict(prediction_file="work_dirs/riskprop_nexar_full/test_predictions.csv")
work_dir = "work_dirs/riskprop_nexar_full"
