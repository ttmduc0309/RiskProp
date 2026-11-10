import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from taa.hooks import AnticipationMetricHook
from taa.metrics import AnticipationMetric, normalized_partial_auc
from tools.analysis_tools.evaluate_nexar_submission import (
    partial_auc_normalized,
)


class TestValidationAUC(unittest.TestCase):
    def test_partial_auc_interpolates_and_handles_ties(self):
        cases = [
            ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.1, 0.5),
            ([0, 1], [0.5, 0.5], 0.1, 0.05),
            ([0, 1], [0.1, 0.9], 0.1, 1.0),
            ([0, 1], [0.9, 0.1], 0.1, 0.0),
            ([0, 1], [0.5, 0.5], 1.0, 0.5),
        ]
        for targets, scores, limit, expected in cases:
            with self.subTest(targets=targets, scores=scores, limit=limit):
                self.assertAlmostEqual(
                    normalized_partial_auc(targets, scores, limit), expected
                )
                self.assertAlmostEqual(
                    partial_auc_normalized(targets, scores, limit), expected
                )
        for limit in (0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                normalized_partial_auc([0, 1], [0.1, 0.9], limit)

    def test_validation_keys_values_and_plot_hook(self):
        # One high-scoring negative lies above both positive videos, so the
        # area below FPR 0.1 is zero, while full AUC remains non-zero.
        predictions = [np.full(30, 0.9), np.full(30, 0.85)]
        predictions.extend(np.full(30, score) for score in [0.95] + [0.1] * 9)
        targets = [True, True] + [False] * 10
        metric = AnticipationMetric(fpr_max=0.1)
        results = metric.calculate(
            predictions, targets, [0] * len(targets), [29] * len(targets), "@"
        )

        self.assertAlmostEqual(results["mAUC_fpr_0.1@"], 0.0)
        self.assertAlmostEqual(results["mAUC_full@"], 0.9)
        for horizon in ("0.0s", "0.5s", "1.0s", "1.5s"):
            self.assertAlmostEqual(results["AUC_fpr_0.1@" + horizon], 0.0)
        self.assertFalse(any("_std" in key or "_norm" in key for key in results))

        with tempfile.TemporaryDirectory() as directory:
            runner = SimpleNamespace(epoch=0, log_dir=directory)
            hook = AnticipationMetricHook()
            hook.after_val_epoch(runner, results)
            self.assertEqual(hook.AUC5_val, [0.0])

    def test_validation_averages_the_three_distinct_horizons(self):
        positive_a = np.full(30, 0.05)
        positive_b = np.full(30, 0.05)
        positive_a[20:25], positive_b[20:25] = 0.8, 0.35
        positive_a[15:20], positive_b[15:20] = 0.8, 0.8
        negatives = [np.full(30, 0.4)] * 5 + [np.full(30, 0.1)] * 5
        predictions = [positive_a, positive_b] + negatives
        targets = [True, True] + [False] * len(negatives)

        result = AnticipationMetric(fpr_max=0.1).calculate(
            predictions, targets, [0] * len(targets), [29] * len(targets), "@"
        )
        self.assertAlmostEqual(result["AUC_fpr_0.1@0.5s"], 0.5)
        self.assertAlmostEqual(result["AUC_fpr_0.1@1.0s"], 1.0)
        self.assertAlmostEqual(result["AUC_fpr_0.1@1.5s"], 0.0)
        self.assertAlmostEqual(result["mAUC_fpr_0.1@"], 0.5)


if __name__ == "__main__":
    unittest.main()
