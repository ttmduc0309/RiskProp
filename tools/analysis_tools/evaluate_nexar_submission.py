#!/usr/bin/env python3
"""Score a complete Nexar submission on Public, Private, and all test videos.

The mAP is the arithmetic mean of average precision across Nexar's three
anticipation groups, matching ``datasets/Nexar/evaluate_submission.py``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", help="CSV with id and score (or id and target) columns")
    parser.add_argument(
        "--solution",
        default=str(Path(__file__).resolve().parents[3] / "datasets/Nexar/solution.csv"),
        help="Nexar solution CSV (default: ../datasets/Nexar/solution.csv)",
    )
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
    print("mAP (All): {:.6f}".format(grouped_map(merged, merged)))


if __name__ == "__main__":
    main()
