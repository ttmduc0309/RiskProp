#!/usr/bin/env python3
"""Export BADAS predictions for RiskProp's Nexar test evaluator.

This runs inference only when explicitly invoked. It writes one finite final
score per video for Nexar mAP and a full temporal JSONL for RiskProp-style mTTA.
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2
import numpy as np


BADAS_FPS = 8.0
BADAS_WINDOW = 16


def sample_timestamps(total_frames, source_fps, target_fps=BADAS_FPS):
    """Reproduce BADAS ``load_full_video_frames`` sampling timestamps."""
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be finite and positive")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    count = int(round((total_frames / source_fps) * target_fps))
    frame_indices = [int(round(i * source_fps / target_fps)) for i in range(count)]
    frame_indices = [index for index in frame_indices if index < total_frames]
    return np.asarray([index / source_fps for index in frame_indices], dtype=np.float64)


def make_prediction_record(video_id, target, scores, timestamps, onset, event, partition):
    """Validate a BADAS frame-score array and build JSON-safe temporal data."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    times = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.size != times.size:
        raise ValueError("BADAS scores and timestamps must be equally sized and non-empty")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
        raise ValueError("timestamps must be finite and sorted")
    if np.any(np.isinf(values)):
        raise ValueError("BADAS scores cannot contain infinity")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("BADAS returned no finite score for video {}".format(video_id))
    if np.any((values[finite] < 0.0) | (values[finite] > 1.0)):
        raise ValueError("BADAS probabilities must be in [0, 1]")
    if bool(int(target)):
        if onset in (None, "") or event in (None, ""):
            raise ValueError("Positive video {} is missing onset/event annotations".format(video_id))
        onset, event = float(onset), float(event)
        if not math.isfinite(onset) or not math.isfinite(event) or event < onset:
            raise ValueError("Invalid onset/event annotations for video {}".format(video_id))
    return {
        "video_id": str(video_id),
        "target": bool(int(target)),
        # JSON has no NaN value; null explicitly marks BADAS's 16-frame warm-up.
        "scores": [float(value) if valid else None for value, valid in zip(values, finite)],
        "timestamps_sec": times.tolist(),
        "onset_time_sec": float(onset) if onset not in (None, "") else None,
        "event_time_sec": float(event) if event not in (None, "") else None,
        "test_partition": partition,
        "model": "BADAS-Open",
        "target_fps": BADAS_FPS,
        "window_size_frames": BADAS_WINDOW,
    }


def final_video_score(record):
    """Use the last observed finite BADAS score as the Nexar submission score."""
    scores = [score for score in record["scores"] if score is not None]
    if not scores:
        raise ValueError("No finite score for video {}".format(record["video_id"]))
    return float(scores[-1])


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_annotations(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["is_test"] == "1"]
    by_id = {row["video_id"].zfill(5): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate test video IDs in annotations")
    return by_id


def _load_video(video_path, target_fps):
    """Decode one video to BADAS's sampling grid, suitable for a worker thread."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    timestamps = sample_timestamps(total_frames, source_fps, target_fps)
    from utils.video import load_full_video_frames

    frames = load_full_video_frames(
        video_path=str(video_path), target_size=(224, 224), target_fps=target_fps
    )
    if len(frames) != len(timestamps):
        raise ValueError(
            "Decoded frame count {} differs from timestamp count {} for {}".format(
                len(frames), len(timestamps), video_path
            )
        )
    return total_frames, source_fps, frames, timestamps


def run(args):
    annotations = _read_annotations(args.annotations)
    solution = {}
    with Path(args.solution).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            video_id = row["id"].zfill(5)
            if video_id in solution:
                raise ValueError("Duplicate video ID in solution: {}".format(video_id))
            solution[video_id] = row
    if set(annotations) != set(solution):
        raise ValueError("Test annotations and Nexar solution IDs do not match")
    if args.num_workers < 1:
        raise ValueError("num_workers must be at least one")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_jsonl = output_dir / ".badas_nexar_temporal.partial.jsonl"

    records = []
    if args.resume and partial_jsonl.is_file():
        with partial_jsonl.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                video_id = str(record["video_id"])
                if video_id not in solution:
                    raise ValueError("Partial results contain an unexpected ID: {}".format(video_id))
                if any(item["video_id"] == video_id for item in records):
                    raise ValueError("Partial results contain duplicate ID: {}".format(video_id))
                if bool(record["target"]) != bool(int(solution[video_id]["target"])):
                    raise ValueError("Partial result label mismatch for {}".format(video_id))
                if record.get("inference_precision") != args.precision:
                    raise ValueError("Resume precision differs from partial results")
                if int(record.get("window_batch_size", -1)) != args.window_batch_size:
                    raise ValueError("Resume batch size differs from partial results")
                records.append(record)
        print("Resuming from {} completed clips.".format(len(records)), flush=True)

    if args.precision == "tf32":
        if not args.device.startswith("cuda"):
            raise ValueError("tf32 inference requires a CUDA device")
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True

    badas_root = Path(args.badas_repo).resolve()
    sys.path.insert(0, str(badas_root / "src"))
    from models.vjepa import VJEPAModel

    checkpoint = Path(args.checkpoint).resolve()
    model = VJEPAModel(
        model_name=args.backbone,
        checkpoint_path=str(checkpoint),
        device=args.device,
        frame_count=BADAS_WINDOW,
        img_size=224,
        window_stride=1,
        window_batch_size=args.window_batch_size,
        target_fps=BADAS_FPS,
        use_sliding_window=True,
        inference_precision="bf16" if args.precision == "bf16" else "fp32",
    )

    progress_state = {"index": 0, "video_id": "", "started": 0.0, "last_update": 0.0}

    def report_window_progress(completed, total):
        now = time.monotonic()
        if (
            completed == 0
            or completed == total
            or now - progress_state["last_update"] >= 5.0
        ):
            elapsed = now - progress_state["started"]
            print(
                "[{}/{}] {} windows {}/{} (clip {:.0f}s)".format(
                    progress_state["index"], len(solution), progress_state["video_id"],
                    completed, total, elapsed
                ),
                flush=True,
            )
            progress_state["last_update"] = now

    model.progress_callback = report_window_progress
    print(
        "Loading BADAS checkpoint on {} for {} Nexar test videos (precision={}, windows/batch={}, decode workers={})...".format(
            args.device, len(solution), args.precision, args.window_batch_size, args.num_workers
        ),
        flush=True,
    )
    model.load()
    print("Model loaded; starting temporal inference.", flush=True)

    prediction_rows = [
        {"id": str(record["video_id"]), "score": final_video_score(record)}
        for record in records
    ]
    completed_ids = {str(record["video_id"]) for record in records}
    video_ids = sorted(solution)
    remaining_ids = [video_id for video_id in video_ids if video_id not in completed_ids]
    executor = ThreadPoolExecutor(max_workers=args.num_workers)
    futures = {}
    next_to_schedule = 0
    partial_mode = "a" if args.resume and partial_jsonl.is_file() else "w"
    partial_stream = partial_jsonl.open(partial_mode, encoding="utf-8")

    def schedule_video(video_index):
        video_id = remaining_ids[video_index]
        video_path = Path(args.data_root) / "test" / (video_id + ".mp4")
        if not video_path.is_file():
            raise FileNotFoundError(str(video_path))
        futures[video_index] = executor.submit(_load_video, video_path, BADAS_FPS)

    try:
        while next_to_schedule < min(args.num_workers, len(remaining_ids)):
            schedule_video(next_to_schedule)
            next_to_schedule += 1

        already_completed = len(records)
        for remaining_index, video_id in enumerate(remaining_ids):
            index = already_completed + remaining_index + 1
            print("[{}/{}] decoding {}".format(index, len(solution), video_id), flush=True)
            video_path = Path(args.data_root) / "test" / (video_id + ".mp4")
            _, _, frames, timestamps = futures.pop(remaining_index).result()
            if next_to_schedule < len(remaining_ids):
                schedule_video(next_to_schedule)
                next_to_schedule += 1
            progress_state.update(
                index=index,
                video_id=video_id,
                started=time.monotonic(),
                last_update=0.0,
            )
            print(
                "[{}/{}] {} decoded ({} frames at {:.1f} fps)".format(
                    index, len(solution), video_id, len(frames), BADAS_FPS
                ),
                flush=True,
            )
            annotation = annotations[video_id]
            scores = model.predict(str(video_path), preloaded_frames=frames)
            if len(scores) != len(timestamps):
                raise ValueError(
                    "BADAS output length {} differs from sampled timestamp length {} for {}"
                    .format(len(scores), len(timestamps), video_id)
                )
            record = make_prediction_record(
                video_id,
                annotation["target"],
                scores,
                timestamps,
                annotation.get("onset_time_sec"),
                annotation.get("event_time_sec"),
                annotation.get("test_partition", ""),
            )
            record["window_batch_size"] = args.window_batch_size
            record["inference_precision"] = args.precision
            record["decode_workers"] = args.num_workers
            if bool(int(annotation["target"])) != bool(int(solution[video_id]["target"])):
                raise ValueError("Test annotation label mismatch for {}".format(video_id))
            records.append(record)
            prediction_rows.append({"id": video_id, "score": final_video_score(record)})
            partial_stream.write(json.dumps(record, allow_nan=False) + "\n")
            partial_stream.flush()
            print(
                "[{}/{}] {} complete ({:.1f}s; score {:.6f})".format(
                    len(records), len(solution), video_id,
                    time.monotonic() - progress_state["started"],
                    prediction_rows[-1]["score"],
                ),
                flush=True,
            )
    finally:
        executor.shutdown(wait=True)
        partial_stream.close()

    csv_path = output_dir / "badas_nexar_submission.csv"
    jsonl_path = output_dir / "badas_nexar_temporal.jsonl"

    def write_csv(path):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("id", "score"))
            writer.writeheader()
            writer.writerows(prediction_rows)

    def write_jsonl(path):
        with path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    _atomic_write(csv_path, write_csv)
    _atomic_write(jsonl_path, write_jsonl)
    partial_jsonl.unlink()
    print("Nexar submission scores: {}".format(csv_path))
    print("Temporal predictions: {}".format(jsonl_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--badas-repo", default="../BADAS-Open")
    parser.add_argument("--checkpoint", default="../BADAS-Open/weights/badas_open.pth")
    parser.add_argument("--backbone", default="facebook/vjepa2-vitl-fpc16-256-ssv2")
    parser.add_argument("--data-root", default="data/nexar-collision-prediction")
    parser.add_argument("--annotations", default="data/nexar-collision-prediction/annotations.csv")
    parser.add_argument("--solution", default="../datasets/Nexar/solution.csv")
    parser.add_argument("--output-dir", default="work_dirs/badas_nexar_test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument(
        "--precision", choices=("fp32", "tf32", "bf16"), default="tf32",
        help="tf32 uses CUDA Tensor Cores with small numerical differences; fp32 is exact baseline",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="CPU workers decoding upcoming videos while the GPU runs inference",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume from per-video predictions in the output directory",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
