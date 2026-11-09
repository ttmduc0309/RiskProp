# RiskProp mTTA evaluation audit

Date: 2026-11-03

## Summary

At the time of this audit, the repository implemented `mTTA@0.1` on the
internal validation split. The latest logged value, `mtta@0.1 = 0.8481 s`, was
from 300 validation videos, not the 1,344-video official Nexar test set.

The current test export is insufficient to compute mTTA: it stores one score
per video rather than the temporal score sequence. The raw Nexar test metadata
does contain `time_of_alert`, `time_of_event`, and `time_to_accident`, but the
then-current converter treated the source timestamps as clip-relative frame numbers
and clamps them to the last frame. The dataset loader then drops these temporal
labels for test examples. Correct test mTTA therefore needed full temporal
predictions and reconstructed clip-relative timing from the original metadata.

## Implementation updates

The full-timeline test path was added on 2026-11-04. On 2026-11-09 it was
changed to report a single mTTA metric, following TOP's zero-for-miss behavior:
all positive clips remain in the denominator, and a positive with no
observable post-onset alarm contributes zero, including a clip whose onset is
after its last frame. The separate evaluable-only mTTA was removed.

The prepared Nexar metadata has 672 positive test clips; 16 have onset after
the clip by more than 1 ms (8 Public and 8 Private). These cases now remain in
the single mTTA mean as zero if there is no valid alarm interval. No full
benchmark inference has been run for this metric. See
[`docs/nexar-training-setup.md`](../nexar-training-setup.md) for the command.

## Verified protocol

- RiskProp defines mTTA under `FAR <= 0.1`, averages TTA over valid thresholds,
  and only accepts alarms after anomaly onset. It says mTTA is a compatibility
  metric, not a training objective. [RiskProp paper, evaluation protocol and
  Table 1](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)
- RiskProp's published Nexar test set contains 1,344 videos. Test clips are
  approximately 10 seconds long and end 0.5, 1.0, or 1.5 seconds before the
  event. [RiskProp paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf),
  [local Nexar README](../../../datasets/Nexar/README.md)
- TOP defines TTA as time from an alarm frame to the accident. Its protocol
  restricts mean TTA to thresholds whose negative-set FPR is at most lambda;
  missed positive videos receive TTA zero, and alarms before annotated anomaly
  onset are not valid warnings. [TOP NeurIPS paper, Sec. 3](https://openreview.net/pdf/cb138a5bdcc90c810d7b6e56e8cff55572d06680.pdf)
- RiskProp's official GitHub metric uses a descending list of negative-frame
  scores as its threshold sweep, searches for the first score crossing from the
  anomaly-onset index to the accident index, assigns zero when no crossing
  occurs, then averages across positives and threshold candidates under the
  FPR cutoff. [Official RiskProp `taa/metrics.py`](https://github.com/xingyueye5/RiskProp/blob/main/taa/metrics.py)
- The current checkout follows that mTTA calculation in
  [`taa/metrics.py`](../../taa/metrics.py#L151). It computes metrics for
  `is_val` samples, while `is_test` samples only export the last prediction
  score in [`taa/metrics.py`](../../taa/metrics.py#L89).
- Nexar's released test metadata includes `time_of_alert`, `time_of_event`, and
  `time_to_accident` in each positive split's metadata. [Local Nexar test
  metadata](../../../datasets/Nexar/test-public/positive/metadata.csv)
- The local preparation script converts those absolute source-video times
  directly to frame indices and clamps them to the 10-second test clip. For
  example, test video `00002` has both event and alert frames stored as frame
  301, the clip's last frame. The loader also sets test `target` to `None` and
  discards the onset/event fields. [`tools/prepare_nexar.py`](../../tools/prepare_nexar.py#L46),
  [`taa/datasets.py`](../../taa/datasets.py#L297)

## Recommendation

To get the closest defensible test mTTA to the paper:

1. Keep the official metric's 5-frame causal input snippet and emit one risk
   score every 0.1 seconds across the full 10-second test clip. The present
   `num_clips=30` sampler covers only about 3.4 seconds; full-clip evaluation
   needs a variable-length or padded/masked sliding-window sequence.
2. Save each test video's full score sequence and corresponding output-frame
   timestamps. Do not reduce it to `max(pred[-1])`.
3. Reconstruct relative timing for positives from Nexar metadata. For clip
   duration `D`, event time `t_event`, and `time_to_accident` `h`, estimate the
   clip start as `t_event - h - D`; onset is `time_of_alert - clip_start` and
   the accident is `h` seconds after the final observed frame. Validate this
   relation against the video frame count/FPS and Nexar's time-to-accident map
   before scoring. Prefer actual frame timestamps for the calculation; retain
   the 10-Hz quantized calculation as a reproduction comparison.
4. At each threshold, calculate empirical FAR on negative frame scores. For
   each positive video, find the first threshold crossing at or after onset;
   compute TTA from that output timestamp to the event time, or zero if no
   valid alarm occurs. Average over positive videos and then over thresholds
   satisfying FAR <= 0.1.
5. For direct comparability with released RiskProp code, report its exact
   threshold-grid convention as the primary `mTTA_paper` result. Also report a
   tie-safe sensitivity value using unique thresholds and actual empirical FAR
   filtering: repeated negative scores can make a rank-selected threshold
   exceed the nominal 0.1 FAR.
6. Keep mTTA out of checkpoint selection. The paper explicitly describes it
   as a compatibility metric; select checkpoints using a validation metric
   chosen independently.

## Result interpretation

The existing validation `mtta@0.1 = 0.8481 s` is useful as an internal
diagnostic. It is not the paper's Nexar test result (0.958 s): the validation
split comes from training videos, is sampled only near each collision, and the
test pipeline currently exports no temporal scores. Do not compare these
numbers as if they used the same split and timeline.

