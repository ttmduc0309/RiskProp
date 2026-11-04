"""Timestamp-based mean time-to-accident evaluation helpers."""

from typing import Dict, List, Sequence

import numpy as np

TIMESTAMP_TOLERANCE_SEC = 1e-3


def calculate_mtta(
    predictions: Sequence[Sequence[float]],
    timestamps_sec: Sequence[Sequence[float]],
    targets: Sequence[bool],
    onset_times_sec: Sequence[float],
    event_times_sec: Sequence[float],
    far_max: float = 0.1,
) -> Dict[str, object]:
    """Calculate mTTA using descending negative-score thresholds.

    This follows RiskProp's released threshold-grid convention: every negative
    frame score is a candidate threshold, candidates are ordered descending,
    and the first ``floor(far_max * n_negative_scores)`` candidates are used.
    A positive video with no valid alarm receives TTA zero in ``mtta``.
    ``mtta_evaluable`` excludes only positives whose onset is later than the
    final observed timestamp; misses among observable positives remain zero.
    """
    if not 0 < far_max <= 1:
        raise ValueError("far_max must be in (0, 1]")
    if not (len(predictions) == len(timestamps_sec) == len(targets)):
        raise ValueError("predictions, timestamps_sec, and targets must have equal lengths")
    if len(onset_times_sec) != len(predictions) or len(event_times_sec) != len(predictions):
        raise ValueError("onset and event times must have one value per video")

    scores = [np.asarray(row, dtype=np.float64).reshape(-1) for row in predictions]
    times = [np.asarray(row, dtype=np.float64).reshape(-1) for row in timestamps_sec]
    for score, timestamp in zip(scores, times):
        if score.size == 0 or score.size != timestamp.size:
            raise ValueError("each video needs equally sized, non-empty score and timestamp arrays")
        if not np.all(np.isfinite(score)) or not np.all(np.isfinite(timestamp)):
            raise ValueError("scores and timestamps must be finite")
        if np.any(np.diff(timestamp) < 0):
            raise ValueError("timestamps must be sorted in ascending order")

    negative_scores = np.concatenate([score for score, target in zip(scores, targets) if not target])
    positive_indices = [index for index, target in enumerate(targets) if target]
    if negative_scores.size == 0 or not positive_indices:
        raise ValueError("mTTA requires at least one positive and one negative video")

    # A positive is censored when its annotated onset is beyond the last
    # timestamp the model observed. Metadata rounding within the tolerance is
    # treated as coinciding with the clip endpoint.
    evaluable_positive_indices = [
        index
        for index in positive_indices
        if float(onset_times_sec[index])
        <= float(times[index][-1]) + TIMESTAMP_TOLERANCE_SEC
    ]
    evaluable_positive_set = set(evaluable_positive_indices)
    censored_positive_count = len(positive_indices) - len(evaluable_positive_indices)

    thresholds = np.sort(negative_scores)[::-1]
    tta_per_threshold: List[float] = []
    tta_evaluable_per_threshold: List[float] = []
    for threshold in thresholds:
        video_ttas = []
        evaluable_video_ttas = []
        for index in positive_indices:
            valid = (
                (times[index] >= float(onset_times_sec[index]) - TIMESTAMP_TOLERANCE_SEC)
                & (times[index] <= float(event_times_sec[index]) + TIMESTAMP_TOLERANCE_SEC)
                & (scores[index] >= threshold)
            )
            alarm_indices = np.flatnonzero(valid)
            tta = (
                float(event_times_sec[index] - times[index][alarm_indices[0]])
                if alarm_indices.size
                else 0.0
            )
            tta = max(tta, 0.0)
            video_ttas.append(tta)
            if index in evaluable_positive_set:
                evaluable_video_ttas.append(tta)
        tta_per_threshold.append(float(np.mean(video_ttas)))
        if evaluable_video_ttas:
            tta_evaluable_per_threshold.append(float(np.mean(evaluable_video_ttas)))

    # Match the official code's index-based FAR cutoff, including its floor.
    # Full-size Nexar has enough negative frames that this is non-zero.
    cutoff = int(len(thresholds) * far_max)
    if cutoff == 0:
        raise ValueError("not enough negative frame scores for the requested FAR cutoff")
    selected_thresholds = thresholds[:cutoff]
    empirical_far = [float(np.mean(negative_scores >= threshold)) for threshold in selected_thresholds]
    selected_ttas = tta_per_threshold[:cutoff]
    selected_evaluable_ttas = tta_evaluable_per_threshold[:cutoff]
    mtta_evaluable = float(np.mean(selected_evaluable_ttas)) if selected_evaluable_ttas else None
    return {
        "mtta": float(np.mean(selected_ttas)),
        "mtta_evaluable": mtta_evaluable,
        "tta_by_threshold": selected_ttas,
        "tta_by_threshold_all": tta_per_threshold,
        "tta_by_threshold_evaluable": tta_evaluable_per_threshold,
        "thresholds": selected_thresholds.tolist(),
        "empirical_far": empirical_far,
        "num_positive_videos": len(positive_indices),
        "num_evaluable_positive_videos": len(evaluable_positive_indices),
        "num_censored_positive_videos": censored_positive_count,
        "num_negative_videos": sum(not target for target in targets),
        "num_negative_scores": int(negative_scores.size),
    }
