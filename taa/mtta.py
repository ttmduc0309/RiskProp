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
    A positive video with no observable post-onset alarm receives TTA zero.
    Every positive video remains in the mean, including clips whose onset is
    beyond the final observed timestamp, matching TOP's zero-for-miss behavior.
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

    thresholds = np.sort(negative_scores)[::-1]
    # For a positive video, the first alarm at a threshold is the first time
    # the running maximum score reaches that threshold. Search the monotonic
    # running maximum for all thresholds at once. This avoids a Python loop
    # over every negative frame threshold (which can be tens of thousands on
    # full-timeline test data).
    tta_totals = np.zeros(thresholds.size, dtype=np.float64)
    for index in positive_indices:
        valid = (
            (times[index] >= float(onset_times_sec[index]) - TIMESTAMP_TOLERANCE_SEC)
            & (times[index] <= float(event_times_sec[index]) + TIMESTAMP_TOLERANCE_SEC)
        )
        window_indices = np.flatnonzero(valid)
        if window_indices.size == 0:
            continue

        running_max = np.maximum.accumulate(scores[index][window_indices])
        first_alarm_positions = np.searchsorted(running_max, thresholds, side="left")
        has_alarm = first_alarm_positions < window_indices.size
        video_ttas = np.zeros(thresholds.size, dtype=np.float64)
        alarm_times = times[index][window_indices[first_alarm_positions[has_alarm]]]
        video_ttas[has_alarm] = np.maximum(
            float(event_times_sec[index]) - alarm_times, 0.0
        )
        tta_totals += video_ttas

    tta_per_threshold = (tta_totals / len(positive_indices)).tolist()

    # Match the official code's index-based FAR cutoff, including its floor.
    # Full-size Nexar has enough negative frames that this is non-zero.
    cutoff = int(len(thresholds) * far_max)
    if cutoff == 0:
        raise ValueError("not enough negative frame scores for the requested FAR cutoff")
    selected_thresholds = thresholds[:cutoff]
    empirical_far = [float(np.mean(negative_scores >= threshold)) for threshold in selected_thresholds]
    selected_ttas = tta_per_threshold[:cutoff]
    return {
        "mtta": float(np.mean(selected_ttas)),
        "tta_by_threshold": selected_ttas,
        "tta_by_threshold_all": tta_per_threshold,
        "thresholds": selected_thresholds.tolist(),
        "empirical_far": empirical_far,
        "num_positive_videos": len(positive_indices),
        "num_negative_videos": sum(not target for target in targets),
        "num_negative_scores": int(negative_scores.size),
    }
