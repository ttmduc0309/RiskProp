# Nexar test clips whose alert time is beyond the clip

Date: 2026-11-03

## Summary

This is a real edge case in the local Nexar test metadata, but it is consistent
with the benchmark's intentionally truncated test clips. Nexar says the 1,344
test videos are about 10 seconds long and end 0.5, 1.0, or 1.5 seconds before
the event. The official benchmark computes video-level AP at fixed event
horizons; it does not use `time_of_alert` as its scoring target. Therefore this
edge case does not invalidate official Nexar mAP.

In the local test metadata, **16 of 672 positive clips (2.38%)** have
`time_of_alert` later than the final observed frame by more than 1 ms: 8 Public
and 8 Private. Their alert-onset interval is censored: the video contains no
frames at or after the annotated alert time. The Nexar docs, RiskProp paper,
and released RiskProp evaluator do not document a separate censoring rule for
these samples. The released evaluator falls back to index 0 when it cannot find
the annotated onset/event frame, rather than explicitly marking the case
unobservable. That behavior yields zero TTA in the relevant edge cases, but
the code does not explain whether this was an intentional policy.

## Local data audit

For each positive test row, the converter reconstructs the clip-relative times
from Nexar metadata:

```text
clip_duration = (frame_count - 1) / fps
clip_start    = time_of_event - time_to_accident - clip_duration
alert_in_clip = time_of_alert - clip_start
```

The table below lists rows where `alert_in_clip - clip_duration > 0.001 s`.
The source paths are relative to `datasets/Nexar/`.

| Split | ID | Alert after last frame |
|---|---:|---:|
| Public | 00185 | 0.066 s |
| Public | 00211 | 0.164 s |
| Public | 00258 | 0.366 s |
| Public | 00278 | 0.167 s |
| Public | 00347 | 0.467 s |
| Public | 00419 | 0.016 s |
| Public | 00432 | 0.467 s |
| Public | 00439 | 0.279 s |
| Private | 00012 | 0.239 s |
| Private | 00165 | 0.467 s |
| Private | 00198 | 0.467 s |
| Private | 00241 | 0.239 s |
| Private | 00536 | 0.030 s |
| Private | 00680 | 0.433 s |
| Private | 00746 | 0.266 s |
| Private | 01027 | 0.334 s |

Example: `test-private/positive/00198.mp4` has a last-frame time of about
10.5556 seconds in clip-relative coordinates. Its reconstructed alert time is
about 11.0226 seconds, 0.467 seconds beyond the final frame. Its event is
estimated at 11.0556 seconds, consistent with `time_to_accident = 0.5` seconds.
The MP4 is not expected to show the collision: it is a benchmark clip ending
before the event.

Seven additional positives have alert time equal to the last frame within the
1 ms metadata precision. These are boundary samples, not clear post-clip
alerts. The 16 above are counted separately using a strict >1 ms tolerance.

The public and private files can be located at:

- `datasets/Nexar/test-public/positive/<ID>.mp4`
- `datasets/Nexar/test-private/positive/<ID>.mp4`

The generated annotation table is
[`data/nexar-collision-prediction/annotations.csv`](../../data/nexar-collision-prediction/annotations.csv),
and the reconstruction is implemented in
[`tools/prepare_nexar.py`](../../tools/prepare_nexar.py).

## What the primary sources say

1. Nexar's dataset card states that test clips are approximately 10 seconds
   and end 0.5/1.0/1.5 seconds before the event. It describes the benchmark
   as mAP over pre-event horizons and says the submission contains one score
   per test video. It does not say that every test clip contains the alert
   onset. [Official Nexar dataset card](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction/blob/main/README.md)
2. The Nexar challenge uses AP at event-relative horizons (500, 1000, and
   1500 ms), not `time_of_alert` as the per-frame benchmark label. So an alert
   timestamp after a clip's last frame does not prevent calculating the
   official video-level mAP. [Nexar Kaggle data description](https://www.kaggle.com/competitions/nexar-collision-prediction/data)
3. RiskProp defines mTTA under FAR <= 0.1 and says only alarms after anomaly
   onset count. Its paper does not describe the case where onset is later than
   the end of a Nexar test clip. It reports a Nexar mTTA, but does not specify
   how such censored cases contribute. [RiskProp paper, evaluation protocol](https://arxiv.org/pdf/2603.27165)
4. RiskProp's public implementation computes onset/event indices from
   `frame_inds`; if no frame reaches the given annotation, it sets the index
   to 0. Its TTA calculation then does not explicitly mark an onset-after-clip
   sample as censored. [RiskProp metric source](https://github.com/xingyueye5/RiskProp/blob/main/taa/metrics.py)
5. I found no primary-source issue, protocol note, or official Nexar/RiskProp
   code documenting a special fix for these test cases. The public evaluator
   is designed around video-level predictions and fixed-horizon AP, which is
   why the issue does not arise for the official metric.

## Interpretation and recommendation

This should be described as **right-censoring of alert onset in the supplied
test clip**, not immediately as a wrong label or damaged video. The annotations
are expressed on the original timeline; the test clip deliberately exposes
only a short pre-event suffix. For 16 samples, that suffix ends before the
annotated alert moment.

- For official Nexar mAP, keep the released clips and compute the specified
  horizon-based score. Do not replace test clips with unrelated 40-second
  training videos; that changes the test examples and cannot reproduce the
  benchmark.
- For temporal mTTA, retain the complete frame timeline and distinguish
  **no alarm after observed onset** from **onset never observed because the
  clip ended first**. The latter is censored, not directly measurable from
  this clip. Report the number/rate and, if a sensitivity analysis is useful,
  show both (a) a paper-code-compatible result that assigns these cases zero
  and (b) a clearly named evaluable-only result. The evaluable-only result
  changes the population and must not be presented as the paper-exact result.
- Exact comparability with RiskProp's published Nexar mTTA remains uncertain
  until the authors clarify their treatment or release the experiment script
  and exact test split. The implementation's fallback-to-zero is evidence of
  code behavior, not proof of the authors' intended statistical policy.

## Local implementation note

The local evaluator now returns two test metrics:

- `mtta_test@0.1`: reference-code-compatible; a positive with no valid alarm,
  including a censored onset, contributes zero.
- `mtta_test_evaluable@0.1`: excludes only positive clips whose onset is beyond
  the final observed timestamp. Missed alarms on observable clips still
  contribute zero. The metric is omitted when there are no evaluable positives.

It also reports the total, evaluable, and censored positive counts. This keeps
the code-compatible scalar available while making the censored population
visible. The evaluable-only result is supplementary and should not be presented
as directly paper-comparable.
