import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve

from tools.analysis_tools.evaluate_nexar_submission import (
    grouped_mauc,
    partial_auc_normalized,
    write_combined_report,
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
        expected_standardized = np.mean(
            [
                roc_auc_score(rows.target, rows.score, max_fpr=0.1)
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
        self.assertAlmostEqual(
            result["mean"]["partial_standardized"], expected_standardized
        )
        self.assertAlmostEqual(
            result["mean"]["partial_normalized"], expected_normalized
        )
        self.assertEqual(set(result["by_group"]), {"0", "1"})

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
                            "scores": [score, score],
                            "timestamps_sec": [0.0, 1.0],
                            "onset_time_sec": 0.0 if target else None,
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
            self.assertIn("Standardized partial mAUC", report)
            self.assertIn("Normalized partial mAUC", report)


if __name__ == "__main__":
    unittest.main()
