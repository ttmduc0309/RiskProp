import numpy as np
import unittest
import json
import os
import tempfile

from taa.metrics import AnticipationMetric
from taa.mtta import calculate_mtta, calculate_test_mtta
from taa.transforms import SampleFramesBeforeAccident, SampleFullVideoCausalFrames


class TestNexarMTTA(unittest.TestCase):
    def test_test_mtta_uses_observed_post_onset_alarms_and_event_time(self):
        predictions = [
            [0.99, 0.4, 0.95, 0.4],  # First valid alarm is at 1.2s.
            [0.99, 0.99],             # Onset is after the clip.
            [0.99, 0.1, 0.1],        # Only the pre-onset score crosses.
            [0.1, 0.95, 0.1],        # Alarm exactly at onset.
        ] + [[score] for score in np.linspace(0.9, 0.0, 10)]
        result = calculate_test_mtta(
            predictions=predictions,
            timestamps_sec=[[0.0, 0.8, 1.2, 1.35], [0.0, 1.0],
                            [0.0, 0.5, 1.0], [0.0, 0.3, 0.6]] + [[0.0]] * 10,
            targets=[True] * 4 + [False] * 10,
            onset_times_sec=[0.8, 1.5, 0.5, 0.3] + [None] * 10,
            event_times_sec=[2.0, 2.0, 2.0, 1.5] + [None] * 10,
            far_max=0.1,
        )

        # At threshold 0.9: TTA = [0.8, 0, 0, 1.2] seconds.
        self.assertAlmostEqual(result["mtta"], 0.5)
        self.assertAlmostEqual(result["mtta_onset_observable"], 2.0 / 3.0)
        self.assertEqual(result["num_positive_onset_after_clip"], 1)
        self.assertEqual(result["num_positive_onset_observable"], 3)
        self.assertLessEqual(max(result["empirical_far"]), 0.1)

    def test_test_mtta_never_uses_tied_negative_threshold_with_far_above_limit(self):
        result = calculate_test_mtta(
            predictions=[[0.95, 0.95]] + [[0.9]] * 10,
            timestamps_sec=[[0.0, 0.5]] + [[0.0]] * 10,
            targets=[True] + [False] * 10,
            onset_times_sec=[0.0] + [None] * 10,
            event_times_sec=[1.0] + [None] * 10,
            far_max=0.1,
        )
        self.assertAlmostEqual(result["mtta"], 1.0)
        self.assertEqual(result["empirical_far"], [0.0])

    def test_test_mtta_reports_unavailable_observable_mean(self):
        result = calculate_test_mtta(
            predictions=[[0.99]] + [[0.9]] * 10,
            timestamps_sec=[[0.0]] * 11,
            targets=[True] + [False] * 10,
            onset_times_sec=[0.5] + [None] * 10,
            event_times_sec=[1.0] + [None] * 10,
            far_max=0.1,
        )
        self.assertEqual(result["mtta"], 0.0)
        self.assertIsNone(result["mtta_onset_observable"])
        self.assertEqual(result["num_positive_onset_after_clip"], 1)
        self.assertEqual(result["num_positive_onset_observable"], 0)

    def test_test_mtta_rejects_onset_after_event(self):
        with self.assertRaisesRegex(ValueError, "onset must not be after"):
            calculate_test_mtta(
                predictions=[[0.99]] + [[0.9]] * 10,
                timestamps_sec=[[0.0]] * 11,
                targets=[True] + [False] * 10,
                onset_times_sec=[2.0] + [None] * 10,
                event_times_sec=[1.0] + [None] * 10,
            )

    def test_all_positive_videos_contribute_to_single_mtta(self):
        negative_scores = np.linspace(0.9, 0.0, 10)
        result = calculate_mtta(
            predictions=[
                [0.1, 0.95, 0.2],  # Alarm at frame 1, event at frame 2.
                [0.1, 0.99],       # Event is after the final observed frame.
                [0.1, 0.2],        # No alarm crosses threshold.
            ] + [[score] for score in negative_scores],
            timestamps_sec=[
                [0.0, 0.1, 0.2],
                [0.0, 0.1],
                [0.0, 0.1],
            ] + [[0.0] for _ in negative_scores],
            targets=[True, True, True] + [False] * len(negative_scores),
            event_times_sec=[0.2, 0.2, 0.1] + [0.0] * len(negative_scores),
            far_max=0.1,
        )

        # TOP frame-index behavior: (1 frame / 10 Hz + 0 + 0) / 3.
        self.assertAlmostEqual(result["mtta"], 1.0 / 30.0)
        self.assertEqual(result["num_positive_videos"], 3)

    def test_legacy_frame_index_helper_without_onset_matches_top_old(self):
        result = calculate_mtta(
            predictions=[[0.8, 0.1, 0.9]] + [[score] for score in np.linspace(0.7, 0.0, 10)],
            timestamps_sec=[[0.0, 0.1, 0.2]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            event_times_sec=[0.2] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertAlmostEqual(result["mtta"], 0.2)
        self.assertEqual(result["num_positive_videos"], 1)
        self.assertEqual(result["num_negative_videos"], 10)
        self.assertAlmostEqual(result["empirical_far"][0], 0.1)

    def test_onset_gated_mtta_ignores_alarms_before_onset(self):
        positive_scores = [0.99] + [0.8] * 19
        negative_scores = np.linspace(0.95, 0.0, 10)
        result = calculate_mtta(
            predictions=[positive_scores] + [[score] for score in negative_scores],
            timestamps_sec=[np.arange(20) / 10.0] + [[0.0] for _ in negative_scores],
            targets=[True] + [False] * len(negative_scores),
            event_times_sec=[1.9] + [0.0] * len(negative_scores),
            start_indices=[10] + [0] * len(negative_scores),
            far_max=0.1,
        )

        # The only threshold crossing is at frame 0, before onset frame 10.
        self.assertEqual(result["mtta"], 0.0)

    def test_onset_gated_mtta_matches_top_and_riskprop_main_formula(self):
        predictions = [
            np.array([0.2, 0.95, 0.4, 0.8, 0.7]),
            np.array([0.99, 0.6, 0.85, 0.9, 0.1]),
        ] + [np.array([score, score / 2]) for score in np.linspace(0.8, 0.0, 10)]
        targets = [True, True] + [False] * 10
        onset_indices = [2, 1] + [0] * 10
        event_indices = [4, 4] + [0] * 10
        negative_scores = np.sort(
            np.concatenate([pred for pred, target in zip(predictions, targets) if not target])
        )[::-1]
        top_per_video = [
            (event - onset - np.argmax(pred[onset : event + 1, None] >= negative_scores, axis=0))
            / 10.0
            * np.any(pred[onset : event + 1, None] >= negative_scores, axis=0)
            for pred, target, onset, event in zip(
                predictions, targets, onset_indices, event_indices
            )
            if target
        ]
        expected_by_threshold = np.asarray(top_per_video).mean(axis=0)

        result = calculate_mtta(
            predictions=predictions,
            timestamps_sec=[np.arange(len(pred), dtype=float) / 10.0 for pred in predictions],
            targets=targets,
            event_times_sec=[index / 10.0 for index in event_indices],
            start_indices=onset_indices,
            far_max=0.1,
        )

        np.testing.assert_allclose(result["tta_by_threshold_all"], expected_by_threshold)
        self.assertAlmostEqual(result["mtta"], expected_by_threshold[: int(len(negative_scores) * 0.1)].mean())

    def test_validation_mtta_passes_anomaly_onset_to_tta_calculation(self):
        metric = AnticipationMetric(vis_list={})
        positive = np.array([0.99] + [0.8] * 19)
        negatives = [np.full(20, score) for score in np.linspace(0.95, 0.0, 10)]

        result = metric.calculate(
            preds=[positive] + negatives,
            labels=[True] + [False] * len(negatives),
            abnormal_start_inds=[10] + [0] * len(negatives),
            accident_inds=[19] + [0] * len(negatives),
            sep="@",
        )

        # Validation must not count a high score that happened before onset.
        self.assertEqual(result["mtta@0.1"], 0.0)

    def test_assigns_zero_when_no_alarm_occurs(self):
        result = calculate_mtta(
            predictions=[[0.1, 0.2]] + [[score] for score in np.linspace(0.8, 0.0, 10)],
            timestamps_sec=[[0.0, 0.1]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            event_times_sec=[0.1] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertEqual(result["mtta"], 0.0)

    def test_legacy_frame_index_helper_returns_zero_when_event_is_after_clip(self):
        result = calculate_mtta(
            predictions=[[0.9, 0.95]] + [[score] for score in np.linspace(0.8, 0.0, 10)],
            timestamps_sec=[[0.0, 0.1]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            event_times_sec=[0.2] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertEqual(result["mtta"], 0.0)

    def test_uses_top_zero_index_fallback_when_event_is_after_clip(self):
        result = calculate_mtta(
            predictions=[[0.1, 0.8]] + [[score] for score in np.linspace(0.7, 0.0, 10)],
            timestamps_sec=[[0.0, 0.1]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            event_times_sec=[0.2] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertEqual(result["mtta"], 0.0)

    def test_shared_tta_logic_matches_previous_validation_frame_calculation(self):
        predictions = [np.array([0.1, 0.7, 0.9]), np.array([0.2, 0.8, 0.6])] + [
            np.array([score, score / 2]) for score in np.linspace(0.6, 0.0, 10)
        ]
        targets = [True, True] + [False] * 10
        event_indices = [2, 2] + [0] * 10
        negative_scores = np.sort(np.concatenate([p for p, target in zip(predictions, targets) if not target]))[::-1]
        top_per_video = [
            (event - np.argmax(pred[: event + 1, None] >= negative_scores, axis=0))
            / 10.0
            * np.any(pred[: event + 1, None] >= negative_scores, axis=0)
            for pred, target, event in zip(predictions, targets, event_indices)
            if target
        ]
        top_tta_by_threshold = np.asarray(top_per_video).mean(axis=0)

        result = calculate_mtta(
            predictions=predictions,
            timestamps_sec=[np.arange(len(pred), dtype=float) / 10.0 for pred in predictions],
            targets=targets,
            event_times_sec=[index / 10.0 for index in event_indices],
            far_max=0.1,
        )

        np.testing.assert_allclose(result["tta_by_threshold_all"], top_tta_by_threshold)

    def test_sampler_emits_causal_windows_for_each_10hz_step(self):
        result = SampleFullVideoCausalFrames(clip_len=5).transform(
            {"total_frames": 101, "fps": 20.0, "start_index": 0}
        )

        windows = result["frame_inds"].reshape(result["num_clips"], result["clip_len"])
        self.assertEqual(result["num_clips"], 51)
        self.assertEqual(windows.shape, (51, 5))
        self.assertTrue(np.all(windows[:, 1:] >= windows[:, :-1]))
        np.testing.assert_array_equal(windows[:, -1], np.arange(0, 101, 2))
        self.assertAlmostEqual(result["frame_timestamps_sec"][-1], 5.0)

    def test_sampler_includes_the_exact_final_frame_for_non_round_duration(self):
        fps = 29.97
        result = SampleFullVideoCausalFrames(clip_len=5).transform(
            {"total_frames": 101, "fps": fps, "start_index": 0}
        )

        windows = result["frame_inds"].reshape(result["num_clips"], result["clip_len"])
        self.assertEqual(windows[-1, -1], 100)
        self.assertAlmostEqual(result["frame_timestamps_sec"][-1], 100 / fps)

    def test_submission_sampler_uses_video_tail_independent_of_test_label(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=30, test_mode=True)
        common = {
            "total_frames": 1214,
            "fps": 30.6,
            "start_index": 0,
            "accident_frame": None,
            "is_test": True,
        }

        positive = sampler.transform({**common, "target": True})
        negative = sampler.transform({**common, "target": False})

        top_indices = np.concatenate(
            np.arange(30)[:, None] + np.arange(5)[None, :]
        ) * 3
        top_indices += 1213 - top_indices[-1]
        np.testing.assert_array_equal(positive["frame_inds"], negative["frame_inds"])
        np.testing.assert_array_equal(positive["frame_inds"], top_indices)
        self.assertEqual(positive["frame_interval"], 3)

    def test_validation_sampler_still_uses_labels_to_align_windows(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=30, test_mode=True)
        common = {"total_frames": 1214, "fps": 30.0, "start_index": 0, "is_test": False}

        positive = sampler.transform(
            {**common, "target": True, "accident_frame": 585}
        )
        negative = sampler.transform(
            {**common, "target": False, "accident_frame": None}
        )

        self.assertEqual(positive["frame_inds"][-1], 585)
        self.assertEqual(negative["frame_inds"][0], 0)

    def test_validation_sampler_matches_public_riskprop_and_top_for_supported_fps(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=10, test_mode=True)
        for fps in (10, 20, 30):
            common = {
                "total_frames": 500,
                "fps": float(fps),
                "start_index": 1,
                "is_test": False,
            }
            interval = fps // 10
            base = np.concatenate(
                np.arange(10)[:, None] + np.arange(5)[None, :]
            ) * interval
            base_max = base[-1]

            positive = sampler.transform(
                {**common, "target": True, "accident_frame": 301}
            )
            expected_positive = np.maximum(
                base + (301 - 1 - base_max), 0
            ) + 1
            np.testing.assert_array_equal(
                positive["frame_inds"], expected_positive
            )

            negative = sampler.transform(
                {**common, "target": False, "accident_frame": None}
            )
            expected_negative = np.maximum(base, 0) + 1
            np.testing.assert_array_equal(
                negative["frame_inds"], expected_negative
            )

    def test_validation_sampler_matches_public_negative_pre_event_fallback(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=10, test_mode=True)
        common = {
            "total_frames": 100,
            "fps": 20.0,
            "start_index": 0,
            "target": False,
            "accident_frame": 80,
            "is_test": False,
        }

        result = sampler.transform(common)
        base = np.concatenate(np.arange(10)[:, None] + np.arange(5)[None, :]) * 2
        expected = np.maximum(base + (80 - 60 - base[-1]), 0)
        np.testing.assert_array_equal(result["frame_inds"], expected)

    def test_validation_sampler_matches_public_start_and_tail_boundary_rules(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=10, test_mode=True)
        common = {"fps": 30.0, "start_index": 0, "is_test": False}

        negative = sampler.transform(
            {
                **common,
                "total_frames": 500,
                "target": False,
                "accident_frame": 450,
            }
        )
        unknown = sampler.transform(
            {
                **common,
                "total_frames": 500,
                "target": None,
                "accident_frame": None,
            }
        )
        short_negative = sampler.transform(
            {
                **common,
                "total_frames": 5,
                "target": False,
                "accident_frame": None,
            }
        )

        base = np.concatenate(np.arange(10)[:, None] + np.arange(5)[None, :]) * 3
        expected_start = np.maximum(base, 0)
        expected_tail = np.maximum(base + (499 - base[-1]), 0)
        expected_short_tail = np.maximum(base + (4 - base[-1]), 0)
        np.testing.assert_array_equal(negative["frame_inds"], expected_start)
        np.testing.assert_array_equal(unknown["frame_inds"], expected_tail)
        np.testing.assert_array_equal(short_negative["frame_inds"], expected_short_tail)

    def test_validation_sampler_matches_public_error_for_missing_positive_event_frame(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=10, test_mode=True)
        with self.assertRaises(TypeError):
            sampler.transform(
                {
                    "total_frames": 100,
                    "fps": 30.0,
                    "start_index": 0,
                    "target": True,
                    "accident_frame": None,
                    "is_test": False,
                }
            )

    def test_validation_sampler_handles_fractional_fps(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=10, test_mode=True)
        offsets = np.concatenate(np.arange(10)[:, None] + np.arange(5)[None, :])
        for fps in (23.6, 25.0, 29.9, 30.5, 30.6):
            with self.subTest(fps=fps):
                base = np.rint(offsets * fps / 10).astype(np.int32)
                common = {"total_frames": 500, "fps": fps, "start_index": 1, "is_test": False}
                positive = sampler.transform({**common, "target": True, "accident_frame": 301})
                negative = sampler.transform({**common, "target": False, "accident_frame": None})
                np.testing.assert_array_equal(positive["frame_inds"], base + 301 - base[-1])
                np.testing.assert_array_equal(negative["frame_inds"], base + 1)
                self.assertEqual(positive["frame_inds"][-1], 301)
                self.assertLessEqual(np.max(np.abs(base - offsets * fps / 10)), 0.5)

    def test_validation_sampler_rejects_invalid_fps(self):
        sampler = SampleFramesBeforeAccident(clip_len=5, num_clips=10, test_mode=True)
        for fps in (0, 9.9, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                sampler.transform({"total_frames": 100, "fps": fps, "start_index": 0,
                                   "target": False, "accident_frame": None, "is_test": False})

    def test_evaluator_reports_test_mtta_and_writes_temporal_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metric = AnticipationMetric(
                prediction_file=os.path.join(temp_dir, "submission.csv"),
                temporal_prediction_file=os.path.join(temp_dir, "temporal.jsonl"),
                compute_test_mtta=True,
                vis_list={},
            )
            results = [
                {
                    "video_id": "positive",
                    "target": True,
                    "pred": np.array([0.1, 0.8, 0.9]),
                    "frame_timestamps_sec": np.array([0.0, 0.1, 0.2]),
                    "onset_time_sec": 0.1,
                    "event_time_sec": 0.5,
                    "test_partition": "Public",
                    "is_val": False,
                    "is_test": True,
                },
                {
                    "video_id": "censored-positive",
                    "target": True,
                    "pred": np.array([0.9, 0.95]),
                    "frame_timestamps_sec": np.array([0.0, 0.1]),
                    "onset_time_sec": 0.3,
                    "event_time_sec": 0.5,
                    "test_partition": "Private",
                    "is_val": False,
                    "is_test": True,
                },
            ]
            results.extend(
                {
                    "video_id": "negative-{}".format(index),
                    "target": False,
                    "pred": np.array([score]),
                    "frame_timestamps_sec": np.array([0.0]),
                    "onset_time_sec": 0.0,
                    "event_time_sec": 0.0,
                    "test_partition": "Private",
                    "is_val": False,
                    "is_test": True,
                }
                for index, score in enumerate(np.linspace(0.7, 0.0, 10))
            )

            computed = metric.compute_metrics(results)
            self.assertAlmostEqual(computed["mtta_test@0.1"], 0.2)
            self.assertEqual(computed["mtta_test_num_positive"], 2)
            self.assertAlmostEqual(computed["mtta_test_onset_observable@0.1"], 0.4)
            self.assertEqual(computed["mtta_test_num_onset_after_clip"], 1)
            self.assertEqual(computed["mtta_test_num_onset_observable"], 1)
            with open(os.path.join(temp_dir, "temporal.jsonl"), encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(len(records), 12)
            self.assertEqual(records[0]["timestamps_sec"], [0.0, 0.1, 0.2])

    def test_evaluator_rejects_missing_positive_onset(self):
        metric = AnticipationMetric(compute_test_mtta=True, vis_list={})
        results = [
            {
                "video_id": "censored-positive",
                "target": True,
                "pred": np.array([0.9, 0.95]),
                "frame_timestamps_sec": np.array([0.0, 1.0]),
                "event_time_sec": 2.0,
                "is_val": False,
                "is_test": True,
            }
        ]
        results.extend(
            {
                "video_id": "negative-{}".format(index),
                "target": False,
                "pred": np.array([score]),
                "frame_timestamps_sec": np.array([0.0]),
                "event_time_sec": 0.0,
                "is_val": False,
                "is_test": True,
            }
            for index, score in enumerate(np.linspace(0.8, 0.0, 10))
        )

        with self.assertRaisesRegex(ValueError, "onset and event"):
            metric.compute_metrics(results)
