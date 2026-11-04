import numpy as np
import unittest
import json
import os
import tempfile

from taa.metrics import AnticipationMetric
from taa.mtta import calculate_mtta
from taa.transforms import SampleFullVideoCausalFrames


class TestNexarMTTA(unittest.TestCase):
    def test_returns_reference_and_censor_aware_mtta(self):
        negative_scores = np.linspace(0.9, 0.0, 10)
        result = calculate_mtta(
            predictions=[
                [0.1, 0.95, 0.2],  # Observable onset; alarm at t=1, event at t=3.
                [0.1, 0.99],       # Onset is after the final observed frame.
                [0.1, 0.2],        # Observable onset, but no alarm crosses threshold.
            ] + [[score] for score in negative_scores],
            timestamps_sec=[
                [0.0, 1.0, 2.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ] + [[0.0] for _ in negative_scores],
            targets=[True, True, True] + [False] * len(negative_scores),
            onset_times_sec=[0.0, 1.5, 0.0] + [0.0] * len(negative_scores),
            event_times_sec=[3.0, 2.0, 2.0] + [0.0] * len(negative_scores),
            far_max=0.1,
        )

        # At the only selected threshold (0.9), reference mTTA includes all
        # positives: (2 seconds + censored zero + missed-alarm zero) / 3.
        self.assertAlmostEqual(result["mtta"], 2.0 / 3.0)
        # The evaluable-only variant excludes just the censored positive; the
        # observable miss remains in the mean as zero: (2 seconds + 0) / 2.
        self.assertAlmostEqual(result["mtta_evaluable"], 1.0)
        self.assertEqual(result["num_positive_videos"], 3)
        self.assertEqual(result["num_evaluable_positive_videos"], 2)
        self.assertEqual(result["num_censored_positive_videos"], 1)

    def test_uses_first_post_onset_alarm_and_event_timestamp(self):
        result = calculate_mtta(
            predictions=[[0.1, 0.8, 0.9]] + [[score] for score in np.linspace(0.7, 0.0, 10)],
            timestamps_sec=[[0.0, 1.0, 2.0]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            onset_times_sec=[0.5] + [0.0] * 10,
            event_times_sec=[3.0] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertAlmostEqual(result["mtta"], 2.0)
        self.assertEqual(result["num_positive_videos"], 1)
        self.assertEqual(result["num_negative_videos"], 10)
        self.assertAlmostEqual(result["empirical_far"][0], 0.1)

    def test_assigns_zero_when_no_alarm_occurs(self):
        result = calculate_mtta(
            predictions=[[0.1, 0.2]] + [[score] for score in np.linspace(0.8, 0.0, 10)],
            timestamps_sec=[[0.0, 1.0]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            onset_times_sec=[0.0] + [0.0] * 10,
            event_times_sec=[2.0] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertEqual(result["mtta"], 0.0)

    def test_evaluable_mtta_is_undefined_when_every_positive_is_censored(self):
        result = calculate_mtta(
            predictions=[[0.9, 0.95]] + [[score] for score in np.linspace(0.8, 0.0, 10)],
            timestamps_sec=[[0.0, 1.0]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            onset_times_sec=[1.5] + [0.0] * 10,
            event_times_sec=[2.0] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertEqual(result["mtta"], 0.0)
        self.assertIsNone(result["mtta_evaluable"])
        self.assertEqual(result["num_evaluable_positive_videos"], 0)
        self.assertEqual(result["num_censored_positive_videos"], 1)

    def test_treats_sub_millisecond_metadata_rounding_as_clip_endpoint(self):
        result = calculate_mtta(
            predictions=[[0.1, 0.8]] + [[score] for score in np.linspace(0.7, 0.0, 10)],
            timestamps_sec=[[0.0, 1.0]] + [[0.0] for _ in range(10)],
            targets=[True] + [False] * 10,
            onset_times_sec=[1.0000003] + [0.0] * 10,
            event_times_sec=[1.5] + [0.0] * 10,
            far_max=0.1,
        )

        self.assertAlmostEqual(result["mtta"], 0.5)
        self.assertAlmostEqual(result["mtta_evaluable"], 0.5)
        self.assertEqual(result["num_evaluable_positive_videos"], 1)
        self.assertEqual(result["num_censored_positive_videos"], 0)

    def test_shared_tta_logic_matches_previous_validation_frame_calculation(self):
        predictions = [np.array([0.1, 0.7, 0.9]), np.array([0.2, 0.8, 0.6])] + [
            np.array([score, score / 2]) for score in np.linspace(0.6, 0.0, 10)
        ]
        targets = [True, True] + [False] * 10
        onset_indices = [1, 0] + [0] * 10
        event_indices = [2, 2] + [0] * 10
        negative_scores = np.sort(np.concatenate([p for p, target in zip(predictions, targets) if not target]))[::-1]
        old_per_video = [
            (event - onset - np.argmax(pred[onset : event + 1, None] >= negative_scores, axis=0))
            / 10.0
            * np.any(pred[onset : event + 1, None] >= negative_scores, axis=0)
            for pred, target, onset, event in zip(predictions, targets, onset_indices, event_indices)
            if target
        ]
        old_tta_by_threshold = np.asarray(old_per_video).mean(axis=0)

        result = calculate_mtta(
            predictions=predictions,
            timestamps_sec=[np.arange(len(pred), dtype=float) / 10.0 for pred in predictions],
            targets=targets,
            onset_times_sec=[index / 10.0 for index in onset_indices],
            event_times_sec=[index / 10.0 for index in event_indices],
            far_max=0.1,
        )

        np.testing.assert_allclose(result["tta_by_threshold_all"], old_tta_by_threshold)

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
                    "frame_timestamps_sec": np.array([0.0, 1.0, 2.0]),
                    "onset_time_sec": 0.5,
                    "event_time_sec": 3.0,
                    "test_partition": "Public",
                    "is_val": False,
                    "is_test": True,
                },
                {
                    "video_id": "censored-positive",
                    "target": True,
                    "pred": np.array([0.9, 0.95]),
                    "frame_timestamps_sec": np.array([0.0, 1.0]),
                    "onset_time_sec": 1.5,
                    "event_time_sec": 2.0,
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
            self.assertAlmostEqual(computed["mtta_test@0.1"], 1.0)
            self.assertAlmostEqual(computed["mtta_test_evaluable@0.1"], 2.0)
            self.assertEqual(computed["mtta_test_num_positive"], 2)
            self.assertEqual(computed["mtta_test_num_evaluable_positive"], 1)
            self.assertEqual(computed["mtta_test_num_censored_positive"], 1)
            self.assertEqual(computed["mtta_test_onset_after_clip"], 1)
            with open(os.path.join(temp_dir, "temporal.jsonl"), encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            self.assertEqual(len(records), 12)
            self.assertEqual(records[0]["timestamps_sec"], [0.0, 1.0, 2.0])

    def test_evaluator_omits_evaluable_metric_when_all_positive_onsets_are_censored(self):
        metric = AnticipationMetric(compute_test_mtta=True, vis_list={})
        results = [
            {
                "video_id": "censored-positive",
                "target": True,
                "pred": np.array([0.9, 0.95]),
                "frame_timestamps_sec": np.array([0.0, 1.0]),
                "onset_time_sec": 1.5,
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
                "onset_time_sec": 0.0,
                "event_time_sec": 0.0,
                "is_val": False,
                "is_test": True,
            }
            for index, score in enumerate(np.linspace(0.8, 0.0, 10))
        )

        computed = metric.compute_metrics(results)

        self.assertEqual(computed["mtta_test@0.1"], 0.0)
        self.assertNotIn("mtta_test_evaluable@0.1", computed)
        self.assertEqual(computed["mtta_test_num_evaluable_positive"], 0)
        self.assertEqual(computed["mtta_test_num_censored_positive"], 1)
