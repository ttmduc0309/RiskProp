"""TOP-style mean time-to-accident evaluation helpers."""

from typing import Dict, List, Optional, Sequence

import numpy as np


def calculate_mtta(
    predictions: Sequence[Sequence[float]],
    timestamps_sec: Sequence[Sequence[float]],
    targets: Sequence[bool],
    event_times_sec: Sequence[float],
    far_max: float = 0.1,
    start_indices: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """Calculate mTTA with the released TOP frame-index convention.

    This follows TOP's released threshold-grid convention: every negative
    frame score is a candidate threshold, candidates are ordered descending,
    and the first ``floor(far_max * n_negative_scores)`` candidates are used.
    When ``start_indices`` is provided, alarms are searched from that frame
    through the accident frame, matching TOP/RiskProp validation mTTA. If it is
    omitted, the full clip is searched as in TOP's older mTTA variant. If the
    accident time is beyond the observed clip, its frame index falls back to
    zero, as in TOP's frame-index code. A missed alarm contributes zero, and
    every positive remains in the mean. Cropped test clips use
    ``calculate_test_mtta`` instead.
    """
    if not 0 < far_max <= 1:
        raise ValueError("far_max must be in (0, 1]")
    if not (len(predictions) == len(timestamps_sec) == len(targets)):
        raise ValueError("predictions, timestamps_sec, and targets must have equal lengths")
    if len(event_times_sec) != len(predictions):
        raise ValueError("event times must have one value per video")
    if start_indices is not None and len(start_indices) != len(predictions):
        raise ValueError("start indices must have one value per video")

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
        event_indices = np.flatnonzero(
            times[index] >= float(event_times_sec[index])
        )
        # TOP maps a missing event frame to index zero.
        event_index = int(event_indices[0]) if event_indices.size else 0
        start_index = int(start_indices[index]) if start_indices is not None else 0
        start_index = min(max(start_index, 0), event_index)
        observed_scores = scores[index][start_index : event_index + 1]
        running_max = np.maximum.accumulate(observed_scores)
        first_alarm_positions = np.searchsorted(running_max, thresholds, side="left")
        has_alarm = first_alarm_positions < observed_scores.size
        video_ttas = np.zeros(thresholds.size, dtype=np.float64)
        alarm_indices = start_index + first_alarm_positions[has_alarm]
        video_ttas[has_alarm] = (event_index - alarm_indices) / 10.0
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


def calculate_test_mtta(
    predictions: Sequence[Sequence[float]],
    timestamps_sec: Sequence[Sequence[float]],
    targets: Sequence[bool],
    onset_times_sec: Sequence[Optional[float]],
    event_times_sec: Sequence[Optional[float]],
    far_max: float = 0.1,
) -> Dict[str, object]:
    """Evaluate observed post-onset alerts against possibly future events.

    Onset and event times are clip-relative metadata, not model inputs. The
    main mean includes every positive, including clips without an observable
    post-onset score. The secondary mean includes only positives with at least
    one scored timestamp between onset and event. Thresholds follow the TOP
    negative-frame rank grid, filtered to satisfy the empirical FAR bound;
    an above-maximum threshold is used if ties leave no valid grid point.
    """
    if not 0 < far_max <= 1:
        raise ValueError("far_max must be in (0, 1]")
    count = len(predictions)
    if not all(len(values) == count for values in (
        timestamps_sec, targets, onset_times_sec, event_times_sec
    )):
        raise ValueError("predictions, timestamps, targets, onset and event times must have equal lengths")

    scores = [np.asarray(row, dtype=np.float64).reshape(-1) for row in predictions]
    times = [np.asarray(row, dtype=np.float64).reshape(-1) for row in timestamps_sec]
    for score, timestamp in zip(scores, times):
        if score.size == 0 or score.size != timestamp.size:
            raise ValueError("each video needs equally sized, non-empty score and timestamp arrays")
        if not np.all(np.isfinite(score)) or not np.all(np.isfinite(timestamp)):
            raise ValueError("scores and timestamps must be finite")
        if np.any(np.diff(timestamp) < 0):
            raise ValueError("timestamps must be sorted in ascending order")

    positive_indices = [index for index, target in enumerate(targets) if target]
    negative_indices = [index for index, target in enumerate(targets) if not target]
    if not positive_indices or not negative_indices:
        raise ValueError("mTTA requires at least one positive and one negative video")

    negative_scores = np.sort(np.concatenate([scores[index] for index in negative_indices]))
    n_negative_scores = len(negative_scores)
    cutoff = int(n_negative_scores * far_max)
    candidates = negative_scores[::-1][:cutoff]
    empirical_far = (n_negative_scores - np.searchsorted(negative_scores, candidates, side="left")) / n_negative_scores
    thresholds = candidates[empirical_far <= far_max]
    empirical_far = empirical_far[empirical_far <= far_max]
    if thresholds.size == 0:
        threshold = np.nextafter(negative_scores[-1], np.inf)
        if not np.isfinite(threshold):
            raise ValueError("cannot construct a finite threshold above the maximum negative score")
        thresholds = np.asarray([threshold])
        empirical_far = np.asarray([0.0])

    totals = np.zeros(thresholds.size, dtype=np.float64)
    observable_count = 0
    onset_after_clip_count = 0
    tolerance = 1e-6
    for index in positive_indices:
        onset = onset_times_sec[index]
        event = event_times_sec[index]
        if onset is None or event is None or not np.isfinite(onset) or not np.isfinite(event):
            raise ValueError("each positive video needs finite onset and event times")
        onset, event = float(onset), float(event)
        if onset > event + tolerance:
            raise ValueError("positive video onset must not be after its event")
        if onset > times[index][-1] + tolerance:
            onset_after_clip_count += 1
        valid = (times[index] >= onset - tolerance) & (times[index] <= event + tolerance)
        if not np.any(valid):
            continue
        observable_count += 1
        observed_scores = scores[index][valid]
        observed_times = times[index][valid]
        positions = np.searchsorted(np.maximum.accumulate(observed_scores), thresholds, side="left")
        has_alarm = positions < observed_scores.size
        totals[has_alarm] += np.maximum(event - observed_times[positions[has_alarm]], 0.0)

    return {
        "mtta": float(np.mean(totals / len(positive_indices))),
        "mtta_onset_observable": float(np.mean(totals / observable_count)) if observable_count else None,
        "num_positive_videos": len(positive_indices),
        "num_positive_onset_observable": observable_count,
        "num_positive_onset_after_clip": onset_after_clip_count,
        "num_negative_videos": len(negative_indices),
        "num_negative_scores": n_negative_scores,
        "thresholds": thresholds.tolist(),
        "empirical_far": empirical_far.tolist(),
    }
