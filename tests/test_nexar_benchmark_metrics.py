import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

from taa.metrics import AnticipationMetric
from tools.analysis_tools.evaluate_nexar_submission import (
    evaluate_temporal_predictions,
    grouped_mauc,
    partial_auc_normalized,
    write_combined_report,
)
from tools.analysis_tools.run_badas_nexar_inference import (
    final_video_score,
    make_prediction_record,
    sample_timestamps,
)


class TestNexarBenchmarkMAUC(unittest.TestCase):
    def test_partial_normalized_auc_integrates_exact_fpr_interval(self):
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        fpr, tpr, _ = roc_curve(targets, scores)
        expected = auc([0.0, 0.0, 0.1], [0.0, 0.5, 0.5]) / 0.1

        self.assertAlmostEqual(partial_auc_normalized(targets, scores, 0.1), expected)
        # The standardized partial AUC uses sklearn's distinct McClish scaling.
        self.assertAlmostEqual(
            roc_auc_score(targets, scores, max_fpr=0.1), 0.7368421052631579
        )
        self.assertNotAlmostEqual(roc_auc_score(targets, scores, max_fpr=0.1), expected)
        self.assertEqual(fpr[0], 0.0)

    def test_grouped_mauc_is_macro_mean_of_per_group_auc(self):
        frame = pd.DataFrame(
            {
                "group": [0, 0, 0, 0, 1, 1, 1, 1],
                "target": [0, 0, 1, 1] * 2,
                "score": [0.1, 0.4, 0.35, 0.8, 0.1, 0.2, 0.7, 0.9],
            }
        )
        result = grouped_mauc(frame, 0.1)

        expected_full = np.mean(
            [
                roc_auc_score(rows.target, rows.score)
                for _, rows in frame.groupby("group")
            ]
        )
        expected_normalized = np.mean(
            [
                partial_auc_normalized(rows.target, rows.score, 0.1)
                for _, rows in frame.groupby("group")
            ]
        )
        self.assertAlmostEqual(result["mean"]["full"], expected_full)
        self.assertAlmostEqual(result["mean"]["mAUC_fpr_0.1"], expected_normalized)
        self.assertAlmostEqual(result["mean"]["mAUC_fpr_0.1"], 0.75)
        self.assertEqual(set(result["mean"]), {"full", "mAUC_fpr_0.1"})
        self.assertEqual(set(result["by_group"]), {"0", "1"})
        self.assertEqual(set(result["by_group"]["0"]), {"full", "AUC_fpr_0.1"})

    def test_rejects_invalid_fpr_and_single_class_group(self):
        two_class = pd.DataFrame(
            {"group": [0, 0], "target": [0, 1], "score": [0.1, 0.9]}
        )
        with self.assertRaisesRegex(ValueError, "fpr_max"):
            grouped_mauc(two_class, 0.0)
        with self.assertRaisesRegex(ValueError, "only one class"):
            grouped_mauc(
                pd.DataFrame(
                    {"group": [0, 0], "target": [1, 1], "score": [0.1, 0.9]}
                ),
                0.1,
            )

    def test_normalized_partial_auc_at_fpr_one_is_full_auc(self):
        targets = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        self.assertAlmostEqual(
            partial_auc_normalized(targets, scores, 1.0),
            roc_auc_score(targets, scores),
        )

    def test_report_and_json_include_all_test_subset_mauc(self):
        rows = []
        temporal_records = []
        for usage in ("Public", "Private"):
            for group in range(3):
                for index, (target, score) in enumerate(
                    ((0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8))
                ):
                    video_id = "{}-{}-{}".format(usage, group, index)
                    rows.append(
                        {
                            "id": video_id,
                            "target": target,
                            "Usage": usage,
                            "group": group,
                            "score": score,
                        }
                    )
                    temporal_records.append(
                        {
                            "video_id": video_id,
                            "target": bool(target),
                            "scores": [None, score] if index == 0 else [score, score],
                            "timestamps_sec": [0.0, 1.0],
                            "onset_time_sec": (
                                1.5 if target and usage == "Private" and group == 2 and index == 3
                                else 0.0 if target else None
                            ),
                            "event_time_sec": 2.0 if target else None,
                        }
                    )

        merged = pd.DataFrame(rows)
        expected_all = grouped_mauc(merged, 0.1)["mean"]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            temporal_path = temp / "temporal.jsonl"
            temporal_path.write_text(
                "".join(json.dumps(record) + "\n" for record in temporal_records),
                encoding="utf-8",
            )
            _, mauc, _, results_path, report_path = write_combined_report(
                temp / "output", temp / "predictions.csv", temporal_path, merged, 0.1
            )

            stored = json.loads(results_path.read_text(encoding="utf-8"))
            report = report_path.read_text(encoding="utf-8")
            self.assertAlmostEqual(
                stored["mauc"]["All"]["mean"]["full"], expected_all["full"]
            )
            self.assertEqual(set(stored["mauc"]), {"Public", "Private", "All"})
            self.assertEqual(set(mauc), {"Public", "Private", "All"})
            self.assertIn("Test mAUC", report)
            self.assertAlmostEqual(
                stored["mauc"]["All"]["mean"]["mAUC_fpr_0.1"],
                expected_all["mAUC_fpr_0.1"],
            )
            self.assertIn("mAUC_fpr_0.1", report)
            self.assertNotIn("standardized", report.lower())
            self.assertIn("Warm-up frames without scores (excluded) | 6", report)
            self.assertIn("mTTA with observable onset", report)
            self.assertEqual(stored["mtta"]["num_positive_onset_after_clip"], 1)
            self.assertEqual(stored["mtta"]["num_positive_onset_observable"], 11)
            self.assertAlmostEqual(stored["mtta"]["mtta"], stored["mtta"]["mtta_onset_observable"] * 11 / 12)
            self.assertLessEqual(stored["mtta"]["max_empirical_far"], 0.1)

    def test_cli_reports_one_partial_metric_with_requested_fpr(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            solution = []
            predictions = []
            for usage in ("Public", "Private"):
                for group in range(3):
                    for target in (0, 1):
                        video_id = f"{usage}-{group}-{target}"
                        solution.append({"id": video_id, "Usage": usage,
                                         "group": group, "target": target})
                        predictions.append({"id": video_id, "score": 0.9 if target else 0.1})
            solution_path = tmp / "solution.csv"
            prediction_path = tmp / "prediction.csv"
            pd.DataFrame(solution).to_csv(solution_path, index=False)
            pd.DataFrame(predictions).to_csv(prediction_path, index=False)
            script = Path(__file__).resolve().parents[1] / "tools/analysis_tools/evaluate_nexar_submission.py"
            completed = subprocess.run(
                [sys.executable, str(script), str(prediction_path), "--solution",
                 str(solution_path), "--far-max", "0.2"],
                check=True, text=True, capture_output=True,
            )
            self.assertEqual(completed.stdout.count("mAUC_fpr_0.2: 1.000000"), 3)
            self.assertNotIn("standardized", completed.stdout.lower())
            self.assertNotIn("normalized", completed.stdout.lower())

    def test_badas_timestamps_match_its_frame_sampling_rule(self):
        times = sample_timestamps(total_frames=30, source_fps=30.0, target_fps=8.0)
        expected_indices = [0, 4, 8, 11, 15, 19, 22, 26]
        np.testing.assert_allclose(times, np.asarray(expected_indices) / 30.0)

    def test_badas_warmup_is_json_null_and_final_score_uses_last_finite_point(self):
        record = make_prediction_record(
            "00123", 1, [np.nan, np.nan, 0.2, 0.7],
            [0.0, 0.125, 0.25, 0.375], 0.2, 1.0, "Public"
        )
        self.assertEqual(record["scores"], [None, None, 0.2, 0.7])
        self.assertEqual(final_video_score(record), 0.7)

    def test_temporal_evaluator_excludes_null_warmup_scores_and_checks_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temporal.jsonl"
            rows = [
                {"video_id": "pos", "target": True, "scores": [None, 0.9],
                 "timestamps_sec": [0.0, 1.0], "onset_time_sec": 0.0,
                 "event_time_sec": 2.0},
                {"video_id": "neg", "target": False, "scores": [0.8, 0.1],
                 "timestamps_sec": [0.0, 1.0], "onset_time_sec": None,
                 "event_time_sec": None},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            result = evaluate_temporal_predictions(
                path, {"pos", "neg"}, {"pos": True, "neg": False}, 0.5
            )
            self.assertEqual(result["num_unscored_frames"], 1)
            self.assertEqual(result["num_negative_scores"], 2)
            self.assertAlmostEqual(result["mtta"], 1.0)
            with self.assertRaisesRegex(ValueError, "labels do not match"):
                evaluate_temporal_predictions(
                    path, {"pos", "neg"}, {"pos": False, "neg": True}, 0.5
                )

    def test_direct_and_report_test_mtta_agree_on_the_same_predictions(self):
        records = [
            {"video_id": "observed", "target": True, "scores": [0.99, 0.95],
             "timestamps_sec": [0.0, 1.0], "onset_time_sec": 0.5, "event_time_sec": 1.5},
            {"video_id": "censored", "target": True, "scores": [0.99, 0.99],
             "timestamps_sec": [0.0, 1.0], "onset_time_sec": 1.2, "event_time_sec": 1.5},
        ] + [
            {"video_id": "neg-{}".format(i), "target": False, "scores": [float(score)],
             "timestamps_sec": [0.0], "onset_time_sec": None, "event_time_sec": None}
            for i, score in enumerate(np.linspace(0.9, 0.0, 10))
        ]
        metric_results = [
            {**record, "pred": np.asarray(record["scores"]),
             "frame_timestamps_sec": np.asarray(record["timestamps_sec"]),
             "is_val": False, "is_test": True}
            for record in records
        ]
        direct = AnticipationMetric(compute_test_mtta=True, vis_list={}).compute_metrics(metric_results)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "temporal.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            expected_targets = {row["video_id"]: row["target"] for row in records}
            report = evaluate_temporal_predictions(path, set(expected_targets), expected_targets, 0.1)

        self.assertAlmostEqual(direct["mtta_test@0.1"], 0.25)
        self.assertAlmostEqual(direct["mtta_test@0.1"], report["mtta"])
        self.assertAlmostEqual(direct["mtta_test_onset_observable@0.1"], report["mtta_onset_observable"])
        self.assertEqual(direct["mtta_test_num_onset_after_clip"], report["num_positive_onset_after_clip"])

    def test_badas_adapter_rejects_shape_mismatch_infinite_and_all_nan(self):
        with self.assertRaisesRegex(ValueError, "equally sized"):
            make_prediction_record("x", 0, [0.5], [0.0, 1.0], None, None, "")
        with self.assertRaisesRegex(ValueError, "cannot contain infinity"):
            make_prediction_record("x", 0, [np.inf], [0.0], None, None, "")
        with self.assertRaisesRegex(ValueError, "no finite score"):
            make_prediction_record("x", 0, [np.nan], [0.0], None, None, "")
        with self.assertRaisesRegex(ValueError, "in \[0, 1\]"):
            make_prediction_record("x", 0, [1.1], [0.0], None, None, "")
        with self.assertRaisesRegex(ValueError, "missing onset/event"):
            make_prediction_record("x", 1, [0.5], [0.0], None, None, "")


if __name__ == "__main__":
    unittest.main()
