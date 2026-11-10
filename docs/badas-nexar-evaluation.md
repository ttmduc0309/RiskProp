# BADAS evaluation on Nexar with RiskProp metrics

This adapter runs BADAS inference over the Nexar test videos and writes inputs
for `tools/analysis_tools/evaluate_nexar_submission.py`. It does not change
BADAS weights, training, or preprocessing.

## Input assumptions

- `data/nexar-collision-prediction/annotations.csv` is the enriched RiskProp
  annotation file, including all test video IDs, labels, test partitions,
  `onset_time_sec`, and `event_time_sec`.
- Test videos are available as `data/nexar-collision-prediction/test/NNNNN.mp4`.
- The BADAS checkout and `weights/badas_open.pth` exist locally. Its V-JEPA2
  backbone and processor must also be available through the Hugging Face cache
  or network access.
- Run from the RiskProp repository root in an environment with BADAS's model
  dependencies, plus OpenCV, NumPy, pandas, and scikit-learn.

## Run inference and evaluate

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n badas --no-capture-output \
  python tools/analysis_tools/run_badas_nexar_inference.py \
  --badas-repo ../BADAS-Open \
  --checkpoint ../BADAS-Open/weights/badas_open.pth \
  --data-root data/nexar-collision-prediction \
  --annotations data/nexar-collision-prediction/annotations.csv \
  --solution ../datasets/Nexar/solution.csv \
  --output-dir work_dirs/badas_nexar_test \
  --window-batch-size 4 --precision tf32 --num-workers 2

conda run -n riskprop python tools/analysis_tools/evaluate_nexar_submission.py \
  work_dirs/badas_nexar_test/badas_nexar_submission.csv \
  --solution ../datasets/Nexar/solution.csv \
  --temporal-predictions work_dirs/badas_nexar_test/badas_nexar_temporal.jsonl \
  --output-dir work_dirs/badas_nexar_test/metrics
```

The first command performs model inference on GPU 0 and streams progress to the
terminal. It reports the current video and sliding-window progress at least
every five seconds. Two CPU workers predecode upcoming videos while the single
GPU process runs the current clip. The second command uses the RiskProp environment to compute grouped Nexar
mAP (Public, Private, All), full mAUC, the normalized partial mAUC
`mAUC_fpr_0.1` (ROC area through FPR 0.1 divided by 0.1), and full-timeline
mTTA at FAR 0.1. It writes JSON and Markdown results.

If inference is interrupted, per-video predictions remain in
`.badas_nexar_temporal.partial.jsonl`. Rerun the same inference command with
`--resume` to skip completed videos and continue. A fresh run without
`--resume` starts a new partial file.

## Score and timing conventions

- BADAS emits frame-level probabilities on an 8 Hz grid. The adapter recreates
  BADAS's own OpenCV frame sampling rule and stores timestamps in source-video
  seconds; it does not relabel the 8 Hz scores as 10 Hz.
- BADAS's 16-frame predictive warm-up is unscored. The JSONL stores these as
  `null`; mTTA excludes those score/timestamp pairs instead of treating them as
  zero risk. The report counts excluded warm-up frames. It does not interpolate
  or extrapolate predictions.
- The inference script defaults to TF32 and windows-per-forward batch size 4.
  On one 81-point clip, batch 4 plus TF32 was about 1.26x faster than FP32,
  with maximum absolute score difference `0.000526`. The chosen precision and
  batch size are recorded in the temporal file and final metric report. Use
  `--precision fp32` for the strict FP32 baseline. `--precision bf16` is
  available but was not selected: its maximum score difference on that clip
  was materially larger (`0.0284`).
- Two CPU workers predecode upcoming clips so decoding can overlap GPU
  inference. This avoids running duplicate model copies on GPU. The
  `--num-workers` setting is recorded with the temporal predictions and metric
  report.
- The single score used for Nexar's three-group AP is the last finite temporal
  probability in each clip. This is an explicit video-level submission rule;
  it does not claim BADAS was trained using RiskProp's horizon sampling.
- mTTA uses the remaining observed temporal probabilities and the Nexar
  onset/event annotations. The report includes both RiskProp-code-compatible
  mTTA (censored positives contribute zero) and an evaluable-positive
  sensitivity. Positive clips whose onset occurs after the observed clip end
  cannot produce an observable pre-onset alert.
- The FPR-limited mAUC uses ROC area divided by the FPR limit. Use the
  separately documented RiskProp paper/code audit before describing this
  value as an exact paper reproduction.

No full test-set inference is part of unit tests. They test timestamp
reconstruction, warm-up/null handling, label/ID validation, and the existing
metric calculations with small synthetic inputs.
