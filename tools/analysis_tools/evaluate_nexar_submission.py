#!/usr/bin/env python3
"""Score a complete Nexar submission on Public, Private, and all test videos.

The mAP is the arithmetic mean of average precision across Nexar's three
anticipation groups, matching ``datasets/Nexar/evaluate_submission.py``.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from taa.mtta import calculate_test_mtta  # noqa: E402


def grouped_map(solution, submission):
    """Return mean AP across the groups present in ``solution``."""
    return float(
        np.mean(
            [
                average_precision_score(
                    solution.loc[solution["group"] == group, "target"],
                    solution.loc[solution["group"] == group, "score"],
                )
                for group in sorted(solution["group"].unique())
            ]
        )
    )


def grouped_ap(solution_with_scores):
    return {
        str(group): float(
            average_precision_score(
                rows["target"].astype(int), rows["score"].astype(float)
            )
        )
        for group, rows in solution_with_scores.groupby("group", sort=True)
    }


def partial_auc_normalized(y_true, y_score, fpr_max):
    """Return raw ROC area up to ``fpr_max``, divided by ``fpr_max``.

    The ROC curve is linearly interpolated where it crosses ``fpr_max`` so
    the integration covers exactly [0, fpr_max], independent of threshold
    ties in the predictions.
    """
    if not 0 < fpr_max <= 1:
        raise ValueError("fpr_max must be in (0, 1]")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    stop = int(np.searchsorted(fpr, fpr_max, side="right"))
    clipped_fpr = fpr[:stop].copy()
    clipped_tpr = tpr[:stop].copy()
    if clipped_fpr.size == 0 or clipped_fpr[-1] < fpr_max:
        right = min(stop, len(fpr) - 1)
        left = max(0, right - 1)
        if fpr[right] == fpr[left]:
            interpolated_tpr = tpr[right]
        else:
            ratio = (fpr_max - fpr[left]) / (fpr[right] - fpr[left])
            interpolated_tpr = tpr[left] + ratio * (tpr[right] - tpr[left])
        clipped_fpr = np.append(clipped_fpr, fpr_max)
        clipped_tpr = np.append(clipped_tpr, interpolated_tpr)
    return float(auc(clipped_fpr, clipped_tpr) / fpr_max)


def grouped_mauc(solution_with_scores, fpr_max=0.1):
    """Compute group-averaged ROC metrics for one test subset."""
    if not 0 < fpr_max <= 1:
        raise ValueError("fpr_max must be in (0, 1]")
    group_metrics = {}
    partial_key = "AUC_fpr_{:g}".format(fpr_max)
    for group, rows in solution_with_scores.groupby("group", sort=True):
        targets = rows["target"].astype(int).to_numpy()
        scores = rows["score"].astype(float).to_numpy()
        if not np.all(np.isfinite(scores)):
            raise ValueError("mAUC scores must all be finite")
        if np.unique(targets).size != 2:
            raise ValueError(
                "mAUC requires positive and negative samples in every group; "
                "group {} has only one class".format(group)
            )
        group_metrics[str(group)] = {
            "full": float(roc_auc_score(targets, scores)),
            partial_key: partial_auc_normalized(targets, scores, fpr_max),
        }
    if not group_metrics:
        raise ValueError("Cannot compute mAUC for an empty test subset")
    means = {
        "full": float(np.mean([values["full"] for values in group_metrics.values()])),
        "mAUC_fpr_{:g}".format(fpr_max): float(
            np.mean([values[partial_key] for values in group_metrics.values()])
        ),
    }
    return {"mean": means, "by_group": group_metrics}


def evaluate_temporal_predictions(path, expected_ids, expected_targets, far_max):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            records.append(json.loads(line))
    record_ids = [str(record["video_id"]) for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Temporal predictions contain duplicate video IDs")
    if set(record_ids) != expected_ids:
        raise ValueError("Temporal predictions must exactly match Nexar solution IDs")
    temporal_targets = {
        str(record["video_id"]): bool(record["target"]) for record in records
    }
    if temporal_targets != expected_targets:
        raise ValueError("Temporal prediction labels do not match the Nexar solution")

    predictions = []
    timestamps = []
    skipped_unscored = 0
    for record in records:
        # Predictive models such as BADAS intentionally have no scores for
        # their warm-up frames. JSON null represents those missing values.
        raw_scores = np.asarray(
            [np.nan if score is None else score for score in record["scores"]],
            dtype=np.float64,
        ).reshape(-1)
        raw_timestamps = np.asarray(record["timestamps_sec"], dtype=np.float64).reshape(-1)
        if raw_scores.size != raw_timestamps.size:
            raise ValueError("Temporal scores and timestamps must have equal lengths")
        if not np.all(np.isfinite(raw_timestamps)):
            raise ValueError("Temporal timestamps must all be finite")
        if np.any(np.isinf(raw_scores)):
            raise ValueError("Temporal scores may be finite or null, not infinite")
        valid = np.isfinite(raw_scores)
        if not np.any(valid):
            raise ValueError("Each temporal prediction needs at least one finite score")
        skipped_unscored += int((~valid).sum())
        predictions.append(raw_scores[valid].tolist())
        timestamps.append(raw_timestamps[valid].tolist())

    result = calculate_test_mtta(
        predictions=predictions,
        timestamps_sec=timestamps,
        targets=[bool(record["target"]) for record in records],
        # Nexar negatives intentionally have no onset/event labels.
        onset_times_sec=[
            record.get("onset_time_sec") if record["target"] else None
            for record in records
        ],
        event_times_sec=[
            record.get("event_time_sec") if record["target"] else None
            for record in records
        ],
        far_max=far_max,
    )
    summary = {
        key: result[key]
        for key in (
            "mtta",
            "mtta_onset_observable",
            "num_positive_videos",
            "num_positive_onset_observable",
            "num_positive_onset_after_clip",
            "num_negative_videos",
            "num_negative_scores",
        )
    }
    summary["max_empirical_far"] = max(result["empirical_far"])
    summary["num_thresholds"] = len(result["thresholds"])
    summary["num_unscored_frames"] = skipped_unscored
    summary["inference_precision"] = sorted(
        {str(record.get("inference_precision", "unspecified")) for record in records}
    )
    summary["window_batch_sizes"] = sorted(
        {int(record.get("window_batch_size", 1)) for record in records}
    )
    summary["decode_workers"] = sorted(
        {int(record.get("decode_workers", 1)) for record in records}
    )
    return summary


def write_combined_report(output_dir, prediction_path, temporal_path, merged, far_max):
    group_ap = {
        subset: grouped_ap(
            merged if subset == "All" else merged.loc[merged["Usage"] == subset]
        )
        for subset in ("Public", "Private", "All")
    }
    map_results = {
        subset: float(np.mean(list(group_ap[subset].values())))
        for subset in ("Public", "Private", "All")
    }
    mauc_results = {
        subset: grouped_mauc(
            merged if subset == "All" else merged.loc[merged["Usage"] == subset],
            far_max,
        )
        for subset in ("Public", "Private", "All")
    }
    expected_targets = {
        str(video_id): bool(target)
        for video_id, target in zip(merged["id"], merged["target"])
    }
    mtta = evaluate_temporal_predictions(
        temporal_path, set(expected_targets), expected_targets, far_max
    )
    results = {
        "n_test_videos": int(len(merged)),
        "map": map_results,
        "group_ap": group_ap,
        "mauc": mauc_results,
        "far_max": far_max,
        "mtta": mtta,
        "prediction_csv": str(Path(prediction_path).resolve()),
        "temporal_predictions_jsonl": str(Path(temporal_path).resolve()),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "test_results.json"
    report_path = output_dir / "test_benchmark_report.md"
    results_path.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")

    ap_rows = [
        "| Test subset | Group 0 AP | Group 1 AP | Group 2 AP | Nexar mAP |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset in ("Public", "Private", "All"):
        values = group_ap[subset]
        ap_rows.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                subset, float(values["0"]), float(values["1"]),
                float(values["2"]), map_results[subset]
            )
        )
    partial_key = "mAUC_fpr_{:g}".format(far_max)
    auc_rows = [
        "| Test subset | Full mAUC | {} |".format(partial_key),
        "|---|---:|---:|",
    ]
    for subset in ("Public", "Private", "All"):
        values = mauc_results[subset]["mean"]
        auc_rows.append(
            "| {} | {:.6f} | {:.6f} |".format(
                subset, values["full"], values[partial_key]
            )
        )
    report = """# RiskProp Nexar test evaluation

Test videos: {n}

## Nexar grouped mAP

The official Nexar score averages AP over its three groups.

{ap_table}

## Test mAUC

Each mAUC is the unweighted mean of ROC AUC over the three Nexar groups. The FPR-limited value is the ROC area over `[0, {far:.2f}]`, interpolated at the boundary and divided by `{far:.2f}`.

{auc_table}

## Full-timeline test mTTA

Scores are searched only at observed timestamps from the mapped alert time (onset proxy) through the event time. TTA is the reconstructed event timestamp minus the first observed alarm timestamp; a missed or unobservable alarm contributes zero to the all-positive mean. The event can occur after the final video frame. The secondary mean includes positives with at least one scored timestamp from onset through event. The negative-frame threshold grid is filtered to empirical FAR <= {far:.2f}, including tied scores; if no grid threshold is valid, a threshold above the maximum negative score is used. This is a documented test protocol, not an official Nexar mTTA.

| Metric | Result |
|---|---:|
| mTTA, all positive videos | {mtta:.4f} s |
| mTTA with observable onset | {mtta_observable} |
| Positive videos | {n_pos} |
| Positive videos with observable onset-to-event scores | {n_observable} |
| Positive videos with onset after last scored timestamp | {n_onset_after} |
| Negative videos | {n_neg} |
| Negative frame scores | {n_neg_scores} |
| Selected thresholds | {n_thresholds} |
| Warm-up frames without scores (excluded) | {n_unscored} |
| Inference precision | {precision} |
| Window batch size | {batch_size} |
| CPU decode workers | {decode_workers} |
| Maximum empirical FAR among selected thresholds | {max_far:.6f} |

## Artifacts

- Nexar predictions: `{prediction_csv}`
- Full-timeline predictions: `{temporal_path}`
- Machine-readable results: `{results_path}`
""".format(
        n=len(merged), ap_table="\n".join(ap_rows),
        auc_table="\n".join(auc_rows), far=far_max,
        mtta=mtta["mtta"],
        mtta_observable=("{:.4f} s".format(mtta["mtta_onset_observable"])
                         if mtta["mtta_onset_observable"] is not None else "N/A"),
        n_pos=mtta["num_positive_videos"],
        n_observable=mtta["num_positive_onset_observable"],
        n_onset_after=mtta["num_positive_onset_after_clip"],
        n_neg=mtta["num_negative_videos"],
        n_neg_scores=mtta["num_negative_scores"],
        n_thresholds=mtta["num_thresholds"],
        n_unscored=mtta["num_unscored_frames"],
        precision=", ".join(mtta["inference_precision"]),
        batch_size=", ".join(str(value) for value in mtta["window_batch_sizes"]),
        decode_workers=", ".join(str(value) for value in mtta["decode_workers"]),
        max_far=mtta["max_empirical_far"],
        prediction_csv=Path(prediction_path).resolve(),
        temporal_path=Path(temporal_path).resolve(),
        results_path=results_path,
    )
    report_path.write_text(report, encoding="utf-8")
    return map_results, mauc_results, mtta, results_path, report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="CSV with id and score (or id and target) columns")
    parser.add_argument(
        "--solution",
        default=str(Path(__file__).resolve().parents[3] / "datasets/Nexar/solution.csv"),
        help="Nexar solution CSV (default: ../datasets/Nexar/solution.csv)",
    )
    parser.add_argument(
        "--temporal-predictions",
        help="optional full-timeline JSONL; adds mTTA to a combined report",
    )
    parser.add_argument("--output-dir", help="directory for combined JSON and Markdown results")
    parser.add_argument("--far-max", type=float, default=0.1)
    args = parser.parse_args()

    solution = pd.read_csv(args.solution, dtype={"id": str})
    submission = pd.read_csv(args.submission, dtype={"id": str})
    if "score" not in submission and "target" in submission:
        submission = submission.rename(columns={"target": "score"})
    if not {"id", "score"}.issubset(submission.columns):
        raise ValueError("Submission must contain 'id' and either 'score' or 'target' columns")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicate video IDs")
    if solution["id"].duplicated().any():
        raise ValueError("Solution contains duplicate video IDs")

    merged = solution.merge(
        submission[["id", "score"]], on="id", how="outer", indicator=True, validate="one_to_one"
    )
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    if missing or extra or merged["score"].isna().any():
        raise ValueError(
            "Submission IDs must exactly match the solution (missing={}, extra={})".format(
                missing, extra
            )
        )
    merged["score"] = pd.to_numeric(merged["score"], errors="raise")

    print("Nexar test videos: {}".format(len(merged)))
    for usage in ("Public", "Private"):
        subset = merged.loc[merged["Usage"] == usage]
        print("mAP ({}): {:.6f}".format(usage, grouped_map(subset, subset)))
    all_map = grouped_map(merged, merged)
    print("mAP (All): {:.6f}".format(all_map))
    for subset in ("Public", "Private", "All"):
        subset_rows = merged if subset == "All" else merged.loc[merged["Usage"] == subset]
        auc_values = grouped_mauc(subset_rows, args.far_max)["mean"]
        partial_key = "mAUC_fpr_{:g}".format(args.far_max)
        print("mAUC ({}) full: {:.6f}; {}: {:.6f}".format(
            subset, auc_values["full"], partial_key, auc_values[partial_key]
        ))
    if args.temporal_predictions:
        output_dir = args.output_dir or str(Path(args.submission).resolve().parent)
        map_results, mauc_results, mtta, results_path, report_path = write_combined_report(
            output_dir, args.submission, args.temporal_predictions, merged, args.far_max
        )
        print("mTTA @ FAR {:.2f}: {:.4f}s".format(args.far_max, mtta["mtta"]))
        print("Combined results: {}".format(results_path))
        print("Combined report: {}".format(report_path))


if __name__ == "__main__":
    main()
