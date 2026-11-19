"""Structured per-run metadata and training/validation metric logging."""
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from mmengine.dist import get_rank, get_world_size, is_main_process
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _git_metadata(repo_root, config_path=None):
    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD")
    source_files = list((repo_root / "taa").glob("*.py"))
    source_files.extend(
        [repo_root / "tools" / "train.py", repo_root / "tools" / "smoke_train_val.py"]
    )
    if config_path:
        path = Path(config_path)
        source_files.append(path if path.is_absolute() else Path.cwd() / path)
    source_file_hashes = {}
    for path in sorted(set(source_files)):
        try:
            source_file_hashes[str(path.resolve().relative_to(repo_root.resolve()))] = _sha256_bytes(
                path.read_bytes()
            )
        except (OSError, ValueError):
            continue
    return {
        "repository_root": git("rev-parse", "--show-toplevel"),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(status),
        "changed_paths": [line[3:].strip() for line in status.splitlines()] if status else [],
        "tracked_diff_sha256": _sha256_bytes(diff.encode("utf-8")) if diff is not None else None,
        "source_file_sha256": source_file_hashes,
    }


def _nvidia_smi_metadata():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        return None


def _package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(module_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _system_memory_total():
    try:
        with open("/proc/meminfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _dataset_fingerprint(dataloader, project_root):
    if dataloader is None:
        return None
    dataset = dataloader.dataset
    seen = set()
    while not hasattr(dataset, "get_data_info") and id(dataset) not in seen:
        seen.add(id(dataset))
        child = getattr(dataset, "dataset", None)
        if child is None:
            break
        dataset = child

    records = []
    labels = {"positive": 0, "negative": 0, "unknown": 0}
    split_counts = {"train": 0, "validation": 0, "hidden_test": 0}
    for index in range(len(dataset)):
        try:
            info = dataset.get_data_info(index)
        except (AttributeError, IndexError, TypeError, ValueError):
            records.append({"index": index})
            continue
        target = info.get("target")
        if target is None:
            labels["unknown"] += 1
        elif bool(target):
            labels["positive"] += 1
        else:
            labels["negative"] += 1
        if info.get("is_test", False):
            split_counts["hidden_test"] += 1
        elif info.get("is_val", False):
            split_counts["validation"] += 1
        else:
            split_counts["train"] += 1

        filename = info.get("filename")
        file_metadata = None
        if filename:
            path = Path(filename)
            if not path.is_absolute():
                path = project_root / path
            try:
                file_metadata = {"exists": True, "size_bytes": path.stat().st_size}
            except OSError:
                file_metadata = {"exists": False}
        records.append(
            {
                "dataset": info.get("dataset"),
                "video_id": info.get("video_id"),
                "target": target,
                "is_val": info.get("is_val"),
                "is_test": info.get("is_test"),
                "total_frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "filename": str(filename) if filename else None,
                "file": file_metadata,
            }
        )
    manifest = json.dumps(_json_safe(records), sort_keys=True, separators=(",", ":"))
    return {
        "num_samples": len(dataset),
        "label_counts": labels,
        "split_counts": split_counts,
        "sample_manifest_sha256": _sha256_bytes(manifest.encode("utf-8")),
    }


def _annotation_metadata(config, project_root):
    found = []

    def visit(value):
        if isinstance(value, dict):
            data_root = value.get("data_root")
            ann_file = value.get("ann_file")
            if data_root and ann_file:
                root = Path(data_root)
                if not root.is_absolute():
                    root = project_root / root
                path = root / ann_file
                try:
                    content = path.read_bytes()
                    found.append(
                        {"path": str(path), "size_bytes": len(content), "sha256": _sha256_bytes(content)}
                    )
                except OSError:
                    found.append({"path": str(path), "available": False})
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(config)
    unique = {item["path"]: item for item in found}
    return list(unique.values())


@HOOKS.register_module()
class RunDiagnosticsHook(Hook):
    """Write run metadata and structured per-epoch training/validation metrics."""

    priority = "VERY_LOW"

    def __init__(self):
        self.diagnostics_dir = None
        self.project_root = Path(__file__).resolve().parents[1]
        self._train_history_lengths = {}
        self._train_epoch_started = None
        self._val_started = None

    def before_run(self, runner):
        self.diagnostics_dir = Path(runner.log_dir) / "diagnostics"
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        current_gpu = None
        if current_device is not None:
            properties = torch.cuda.get_device_properties(current_device)
            current_gpu = {
                "logical_index": current_device,
                "name": properties.name,
                "memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        rank_metadata = {
            "rank": get_rank(),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "runner_seed": getattr(runner, "seed", None),
            "torch_initial_seed": torch.initial_seed(),
            "torch_cuda_initial_seed": torch.cuda.initial_seed() if torch.cuda.is_available() else None,
            "current_gpu": current_gpu,
        }
        self._write_json(self.diagnostics_dir / f"rank_{get_rank()}_runtime.json", rank_metadata)
        if not is_main_process():
            return
        cfg = runner.cfg
        config_text = getattr(cfg, "pretty_text", "")
        try:
            first_parameter = next(runner.model.parameters())
            model_dtype = str(first_parameter.dtype)
        except (AttributeError, StopIteration):
            model_dtype = None
        gpu_devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                gpu_devices.append(
                    {
                        "logical_index": index,
                        "name": properties.name,
                        "memory_bytes": properties.total_memory,
                        "compute_capability": [properties.major, properties.minor],
                        "multiprocessor_count": properties.multi_processor_count,
                    }
                )

        optim_cfg = cfg.get("optim_wrapper", {}) or {}
        randomness_cfg = cfg.get("randomness", {}) or {}
        default_hooks = cfg.get("default_hooks", {}) or {}
        checkpoint_cfg = default_hooks.get("checkpoint", {}) or {}
        train_cfg = cfg.get("train_cfg", {}) or {}
        metadata = {
            "schema_version": 1,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_name": runner.experiment_name,
            "work_dir": runner.work_dir,
            "log_dir": runner.log_dir,
            "command": sys.argv,
            "config": {
                "file": getattr(cfg, "filename", None),
                "resolved_config_sha256": _sha256_bytes(config_text.encode("utf-8")),
                "max_epochs": getattr(runner, "max_epochs", None),
                "validation_interval": train_cfg.get("val_interval"),
                "checkpoint_interval": checkpoint_cfg.get("interval"),
                "checkpoint_save_best": checkpoint_cfg.get("save_best"),
                "initial_checkpoint": cfg.get("load_from"),
                "model": cfg.get("model"),
            },
            "source": _git_metadata(self.project_root, getattr(cfg, "filename", None)),
            "randomness": {
                "actual_runner_seed": getattr(runner, "seed", None),
                "configured_seed": randomness_cfg.get("seed"),
                "diff_rank_seed": randomness_cfg.get("diff_rank_seed"),
                "deterministic": randomness_cfg.get("deterministic"),
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            "precision": {
                "model_parameter_dtype": model_dtype,
                "optim_wrapper_class": type(runner.optim_wrapper).__name__ if runner.optim_wrapper else None,
                "amp_enabled": "Amp" in type(runner.optim_wrapper).__name__ if runner.optim_wrapper else False,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            },
            "distributed": {
                "launcher": cfg.get("launcher"),
                "rank": get_rank(),
                "world_size": get_world_size(),
                "per_rank_runtime_files": [
                    f"rank_{rank}_runtime.json" for rank in range(get_world_size())
                ],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "training": {
                "local_batch_size": cfg.get("train_dataloader", {}).get("batch_size"),
                "num_workers": cfg.get("train_dataloader", {}).get("num_workers"),
                "optimizer": optim_cfg,
                "param_scheduler": cfg.get("param_scheduler"),
            },
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "memory_total_bytes": _system_memory_total(),
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "mmengine": _package_version("mmengine"),
                "mmcv": _package_version("mmcv"),
                "mmaction2": _package_version("mmaction2"),
                "numpy": np.__version__,
                "torchvision": _module_version("torchvision"),
                "opencv": _module_version("cv2"),
                "decord": _module_version("decord"),
            },
            "hardware": {
                "hostname": platform.node(),
                "gpus_visible_to_process": gpu_devices,
                "nvidia_smi": _nvidia_smi_metadata(),
            },
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "WORLD_SIZE",
                    "RANK",
                    "LOCAL_RANK",
                    "MASTER_ADDR",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "NVIDIA_TF32_OVERRIDE",
                    "PYTORCH_CUDA_ALLOC_CONF",
                    "CUDA_LAUNCH_BLOCKING",
                    "OMP_NUM_THREADS",
                )
                if os.environ.get(key) is not None
            },
        }
        self._write_json(self.diagnostics_dir / "run_metadata.json", metadata)
        runner.logger.info("Run diagnostics will be written to %s", str(self.diagnostics_dir))

    def before_train(self, runner):
        if not is_main_process():
            return
        metadata_path = self.diagnostics_dir / "run_metadata.json"
        if not metadata_path.exists():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        train_loader = runner.train_dataloader
        val_loader = runner.val_dataloader
        world_size = get_world_size()
        accumulation_steps = getattr(runner.optim_wrapper, "accumulative_counts", 1)
        train_batch = getattr(train_loader, "batch_size", None)
        metadata["distributed"]["effective_global_batch_size"] = (
            train_batch * world_size * accumulation_steps if train_batch else None
        )
        metadata["distributed"]["gradient_accumulation_steps"] = accumulation_steps
        metadata["dataloaders"] = {
            "train": self._loader_metadata(train_loader),
            "validation": self._loader_metadata(val_loader) if val_loader else None,
        }
        metadata["datasets"] = {
            "train": _dataset_fingerprint(train_loader, self.project_root),
            "validation": _dataset_fingerprint(val_loader, self.project_root) if val_loader else None,
            "annotation_files": _annotation_metadata(
                {
                    "train": runner.cfg.get("train_dataloader", {}).get("dataset", {}),
                    "validation": runner.cfg.get("val_dataloader", {}).get("dataset", {}),
                },
                self.project_root,
            ),
        }
        self._write_json(metadata_path, metadata)

    def before_train_epoch(self, runner):
        if not is_main_process():
            return
        self._train_epoch_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._train_history_lengths = {
            key: len(history.data[0]) for key, history in runner.message_hub.log_scalars.items()
        }

    def after_train_epoch(self, runner):
        if not is_main_process():
            return
        metrics = {}
        for key, history in runner.message_hub.log_scalars.items():
            values, counts = history.data
            start = self._train_history_lengths.get(key, 0)
            values, counts = values[start:], counts[start:]
            if len(values):
                metrics[key] = float(np.average(values, weights=counts))
        self._append_jsonl(
            self.diagnostics_dir / "training_epoch_metrics.jsonl",
            {
                "epoch": runner.epoch + 1,
                "global_iteration": runner.iter,
                "duration_seconds": (
                    time.perf_counter() - self._train_epoch_started
                    if self._train_epoch_started is not None
                    else None
                ),
                "peak_cuda_memory_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
                ),
                "peak_cuda_memory_reserved_bytes": (
                    torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
                ),
                "metrics": metrics,
            },
        )

    def before_val(self, runner):
        if is_main_process():
            self._val_started = time.perf_counter()

    def after_val_epoch(self, runner, metrics=None):
        if not is_main_process():
            return
        self._append_jsonl(
            self.diagnostics_dir / "validation_metrics.jsonl",
            {
                "epoch": runner.epoch + 1,
                "global_iteration": runner.iter,
                "num_samples": len(runner.val_dataloader.dataset),
                "duration_seconds": (
                    time.perf_counter() - self._val_started if self._val_started is not None else None
                ),
                "metrics": metrics or {},
            },
        )

    @staticmethod
    def _loader_metadata(dataloader):
        if dataloader is None:
            return None
        sampler = getattr(dataloader, "sampler", None)
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        return {
            "num_samples": len(dataloader.dataset),
            "batch_size_per_rank": getattr(dataloader, "batch_size", None),
            "num_workers": getattr(dataloader, "num_workers", None),
            "persistent_workers": getattr(dataloader, "persistent_workers", None),
            "pin_memory": getattr(dataloader, "pin_memory", None),
            "prefetch_factor": getattr(dataloader, "prefetch_factor", None),
            "worker_init_fn": getattr(
                getattr(dataloader, "worker_init_fn", None), "__qualname__", None
            ),
            "sampler_class": type(sampler).__name__ if sampler is not None else None,
            "sampler_seed": getattr(sampler, "seed", None),
            "sampler_shuffle": getattr(sampler, "shuffle", None),
            "sampler_round_up": getattr(sampler, "round_up", None),
            "batch_sampler_class": type(batch_sampler).__name__ if batch_sampler else None,
        }

    @staticmethod
    def _write_json(path, value):
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")
        temporary_path.replace(path)

    @staticmethod
    def _append_jsonl(path, value):
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(_json_safe(value), sort_keys=True) + "\n")
