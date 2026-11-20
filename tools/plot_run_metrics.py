#!/usr/bin/env python3
"""Plot epoch-level training and validation metrics saved by taa.diagnostics."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def slug(value):
    value = value.replace("@", "at")
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def save_plot(output_dir, filename, title, ylabel, series, xlabel="Epoch"):
    """Save one metric chart; series is a list of (label, epochs, values)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, epochs, values in series:
        ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=4, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def metric_series(rows, keys, prefix=""):
    epochs = [row["epoch"] for row in rows]
    series = []
    for key in keys:
        values = [row.get("metrics", {}).get(key) for row in rows]
        if values and all(isinstance(value, (int, float)) for value in values):
            series.append((key, epochs, values))
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    train_path = args.diagnostics_dir / "training_epoch_metrics.jsonl"
    val_path = args.diagnostics_dir / "validation_metrics.jsonl"
    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    if not train_rows or not val_rows:
        raise SystemExit("Training and validation JSONL files must both contain records")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or args.diagnostics_dir.parent.parent.name
    train_epochs = [row["epoch"] for row in train_rows]
    val_epochs = [row["epoch"] for row in val_rows]
    created = []

    train_metrics = sorted({key for row in train_rows for key in row.get("metrics", {})})
    for key in train_metrics:
        values = [row.get("metrics", {}).get(key) for row in train_rows]
        if not values or not all(isinstance(value, (int, float)) for value in values):
            continue
        short_key = key[len("train/"):] if key.startswith("train/") else key
        filename = "train_{}.png".format(slug(short_key))
        save_plot(
            args.output_dir,
            filename,
            "{}: {} by epoch".format(run_name, key),
            key,
            [(key, train_epochs, values)],
        )
        created.append(filename)

    for field, label in (
        ("duration_seconds", "Epoch duration (seconds)"),
        ("peak_cuda_memory_allocated_bytes", "Peak CUDA allocated memory (GiB)"),
        ("peak_cuda_memory_reserved_bytes", "Peak CUDA reserved memory (GiB)"),
    ):
        values = [row.get(field) for row in train_rows]
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        if field.endswith("_bytes"):
            values = [value / (1024 ** 3) for value in values]
        filename = "train_{}.png".format(slug(field))
        save_plot(
            args.output_dir,
            filename,
            "{}: {} by epoch".format(run_name, label),
            label,
            [(label, train_epochs, values)],
        )
        created.append(filename)

    horizon_groups = (
        ("AP@", "Validation AP by horizon", "Average precision", "val_ap_horizons.png"),
        ("AUC@", "Validation AUC by horizon (as logged)", "AUC", "val_auc_horizons.png"),
        ("AUC_full@", "Validation full AUC by horizon", "AUC", "val_full_auc_horizons.png"),
    )
    val_keys = sorted({key for row in val_rows for key in row.get("metrics", {})})
    for prefix, title, ylabel, filename in horizon_groups:
        keys = [key for key in val_keys if key.startswith(prefix)]
        series = metric_series(val_rows, keys)
        if series:
            save_plot(args.output_dir, filename, "{}: {}".format(run_name, title), ylabel, series)
            created.append(filename)

    horizon_prefixes = tuple(prefix for prefix, _, _, _ in horizon_groups)
    for key in val_keys:
        if key.startswith(horizon_prefixes) or key == "num_samples@":
            continue
        values = [row.get("metrics", {}).get(key) for row in val_rows]
        if not values or not all(isinstance(value, (int, float)) for value in values):
            continue
        filename = {
            "mAP@": "val_mAP.png",
            "mAUC@": "val_mAUC_full.png",
            "mAUC@0.1": "val_mAUC_fpr_0p1.png",
            "mtta@0.1": "val_mTTA_fpr_0p1.png",
            "threshold@0.10": "val_threshold_fpr_0p1.png",
            "tta@0.01": "val_tta_fpr_0p01.png",
            "tta@0.05": "val_tta_fpr_0p05.png",
            "tta@0.1": "val_tta_fpr_0p1.png",
            "tta@1": "val_tta_fpr_1.png",
        }.get(key, "val_{}.png".format(slug(key)))
        save_plot(
            args.output_dir,
            filename,
            "{}: {} by epoch".format(run_name, key),
            key,
            [(key, val_epochs, values)],
        )
        created.append(filename)

    for field, label in (("duration_seconds", "Validation duration (seconds)"), ("num_samples", "Validation sample count")):
        values = [row.get(field) for row in val_rows]
        if not all(isinstance(value, (int, float)) for value in values):
            continue
        filename = "val_{}.png".format(slug(field))
        save_plot(
            args.output_dir,
            filename,
            "{}: {} by epoch".format(run_name, label),
            label,
            [(label, val_epochs, values)],
        )
        created.append(filename)

    index = [
        "# Training and validation plots",
        "",
        "- Run: `{}`".format(run_name),
        "- Training source: `{}` ({} epochs)".format(train_path, len(train_rows)),
        "- Validation source: `{}` ({} epochs)".format(val_path, len(val_rows)),
        "- Each PNG is a separate plot. Validation metric names follow the JSONL keys.",
        "",
        "## Files",
        "",
    ]
    index.extend("- [{}]({})".format(filename, filename) for filename in sorted(created))
    (args.output_dir / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print("Saved {} plots to {}".format(len(created), args.output_dir))


if __name__ == "__main__":
    main()
