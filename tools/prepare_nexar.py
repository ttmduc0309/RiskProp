#!/usr/bin/env python
"""Prepare Nexar metadata and zero-copy video links for RiskProp."""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List

from decord import VideoReader, cpu


TARGET_FPS = 10
FIELDS = [
    "video_id",
    "is_test",
    "total_frames",
    "fps",
    "abnormal_start_frame",
    "accident_frame",
    "target",
]


def _link_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_source = os.path.relpath(str(source.resolve()), str(destination.parent.resolve()))
    if destination.is_symlink():
        if os.readlink(str(destination)) == relative_source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError("Refusing to replace non-symlink video: {}".format(destination))
    destination.symlink_to(relative_source)


def _video_info(video_path: Path):
    reader = VideoReader(str(video_path), ctx=cpu(0))
    fps = float(reader.get_avg_fps())
    count = len(reader)
    if count < 1 or fps < TARGET_FPS:
        raise ValueError("Video has no decodable frames: {}".format(video_path))
    return count, fps


def _frame_at(timestamp: str, fps: float, total_frames: int) -> int:
    frame = int(round(float(timestamp) * fps))
    return min(max(frame, 0), total_frames - 1)


def _read_metadata(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


def _record(
    row: Dict[str, str], source: Path, output_root: Path, split: str, label: int, is_test: int
) -> List[object]:
    video_id = Path(row["file_name"]).stem.zfill(5)
    if not video_id.isdigit():
        raise ValueError("Unexpected Nexar video id: {}".format(row["file_name"]))
    video_path = source / row["file_name"]
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))
    total_frames, fps = _video_info(video_path)
    _link_video(video_path, output_root / split / (video_id + ".mp4"))

    alert = row.get("time_of_alert") or ""
    event = row.get("time_of_event") or ""
    alert_frame = _frame_at(alert, fps, total_frames) if label and alert else ""
    event_frame = _frame_at(event, fps, total_frames) if label and event else ""
    target = "" if is_test else label
    return [video_id, is_test, total_frames, "{:.6f}".format(fps), alert_frame, event_frame, target]


def prepare(source_root: Path, output_root: Path) -> Dict[str, int]:
    rows: List[List[object]] = []
    seen = set()
    counts = {"train": 0, "test": 0}

    for category, label in (("positive", 1), ("negative", 0)):
        for row in _read_metadata(source_root / "train" / category / "metadata.csv"):
            record = _record(row, source_root / "train" / category, output_root, "train", label, 0)
            key = ("train", record[0])
            if key in seen:
                raise ValueError("Duplicate Nexar training video id: {}".format(record[0]))
            seen.add(key)
            rows.append(record)
            counts["train"] += 1

    for subset in ("test-public", "test-private"):
        for category, label in (("positive", 1), ("negative", 0)):
            for row in _read_metadata(source_root / subset / category / "metadata.csv"):
                record = _record(row, source_root / subset / category, output_root, "test", label, 1)
                key = ("test", record[0])
                if key in seen:
                    raise ValueError("Duplicate Nexar test video id across subsets: {}".format(record[0]))
                seen.add(key)
                rows.append(record)
                counts["test"] += 1

    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "annotations.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    temporary.replace(output_root / "annotations.csv")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    counts = prepare(args.source_root, args.output_root)
    print(
        "Prepared {} train and {} test videos. Media are symlinked; no frames were duplicated."
        .format(counts["train"], counts["test"])
    )
    print("Annotations: {}".format(args.output_root / "annotations.csv"))


if __name__ == "__main__":
    main()
