"""Controlled two-L40 precision benchmark with raw synchronized timings.

Launch with torchrun --nproc_per_node=2, binding CUDA_VISIBLE_DEVICES to L40s.
This trains a balanced subset for one epoch and evaluates the full validation
split in FP32. It does not modify the full training configuration.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from mmengine.config import Config
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from taa.splits import nexar_val


@HOOKS.register_module()
class PrecisionBenchmarkHook(Hook):
    priority = "LOW"

    def __init__(self, warmup=10, mode="fp32", seed=20261031):
        self.warmup = warmup
        self.mode = mode
        self.seed = seed
        self.rows = []
        self.metrics = {}
        self.ids = []
        self.first_input_hash = None

    def before_train(self, runner):
        assert dist.get_world_size() == 2, "Benchmark requires exactly two GPUs"
        assert "L40" in torch.cuda.get_device_name(), "Only L40 is authorized"
        if self.mode.startswith("bf16"):
            assert torch.cuda.is_bf16_supported()
        self.start = time.perf_counter()
        model = runner.model.module
        self.initial_head = model.cls_head.fc_cls.weight.detach().clone()

    def before_train_epoch(self, runner):
        torch.cuda.synchronize()
        dist.barrier()
        self.epoch_start = self.last_end = time.perf_counter()
        self.measured_peak_allocated_mib = 0.0
        self.measured_peak_reserved_mib = 0.0

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.compute_start = time.perf_counter()
        self.data_wait = self.compute_start - self.last_end
        if batch_idx == 0:
            digest = hashlib.sha256()
            for tensor in data_batch["inputs"]:
                digest.update(tensor.numpy().tobytes())
            self.first_input_hash = digest.hexdigest()
            self.compute_start = time.perf_counter()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if batch_idx >= self.warmup:
            # LoggerHook resets CUDA peak counters at logging intervals. Keep
            # our own maximum across iterations instead of querying at epoch end.
            self.measured_peak_allocated_mib = max(
                self.measured_peak_allocated_mib,
                torch.cuda.max_memory_allocated() / 2**20)
            self.measured_peak_reserved_mib = max(
                self.measured_peak_reserved_mib,
                torch.cuda.max_memory_reserved() / 2**20)
        grad_norm = float(runner.message_hub.get_scalar("train/grad_norm").current())
        losses = {k: float(v) for k, v in outputs.items() if "loss" in k}
        assert math.isfinite(grad_norm), (batch_idx, "nonfinite gradient norm")
        assert all(math.isfinite(v) for v in losses.values()), (batch_idx, losses)
        self.rows.append(dict(
            iteration=batch_idx + 1, data_wait_s=self.data_wait,
            compute_s=now - self.compute_start,
            step_s=now - self.last_end, grad_norm=grad_norm, **losses))
        for sample in data_batch["data_samples"]:
            self.ids.append(str(sample.get("video_id")))
        if batch_idx + 1 == self.warmup:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.window_start = time.perf_counter()
        self.last_end = time.perf_counter()

    def after_train_epoch(self, runner):
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.window_start
        elapsed_tensor = torch.tensor(elapsed, device="cuda", dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        self.window_seconds = elapsed_tensor.item()
        self.epoch_seconds = time.perf_counter() - self.epoch_start
        self.peak_allocated_mib = self.measured_peak_allocated_mib
        self.peak_reserved_mib = self.measured_peak_reserved_mib
        model = runner.model.module
        self.weights_finite = all(
            bool(torch.isfinite(tensor).all())
            for tensor in model.state_dict().values() if tensor.is_floating_point())
        assert self.weights_finite
        self.head_update_max = (model.cls_head.fc_cls.weight - self.initial_head).abs().max().item()

    def after_val_epoch(self, runner, metrics=None):
        self.metrics = {k: float(v) for k, v in (metrics or {}).items()}

    def after_run(self, runner):
        measured = self.rows[self.warmup:]
        result = dict(
            mode=self.mode, seed=self.seed, rank=dist.get_rank(),
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
            cudnn_version=torch.backends.cudnn.version(),
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            optimizer_wrapper=type(runner.optim_wrapper).__name__,
            train_videos=len(runner.train_dataloader.dataset),
            validation_videos=(len(runner.val_loop.dataloader.dataset)
                               if runner.val_loop is not None else 0),
            batch_per_gpu=runner.train_dataloader.batch_size,
            num_workers_per_gpu=runner.train_dataloader.num_workers,
            warmup_iterations=self.warmup, measured_iterations=len(measured),
            measured_window_s=self.window_seconds,
            synchronized_seconds_per_step=self.window_seconds / len(measured),
            global_samples_per_second=len(measured) * 4 / self.window_seconds,
            mean_data_wait_s=statistics.mean(r["data_wait_s"] for r in measured),
            median_compute_s=statistics.median(r["compute_s"] for r in measured),
            training_epoch_s=self.epoch_seconds,
            total_train_and_val_s=time.perf_counter() - self.start,
            peak_train_allocated_mib=self.peak_allocated_mib,
            peak_train_reserved_mib=self.peak_reserved_mib,
            gradients_and_losses_finite=True, weights_finite=self.weights_finite,
            min_grad_norm=min(r["grad_norm"] for r in self.rows),
            max_grad_norm=max(r["grad_norm"] for r in self.rows),
            head_update_max=self.head_update_max,
            first_input_sha256=self.first_input_hash,
            sample_order_sha256=hashlib.sha256("|".join(self.ids).encode()).hexdigest(),
            validation_metrics=self.metrics, iterations=self.rows)
        output = Path(runner.work_dir) / ("result_rank%d.json" % dist.get_rank())
        output.write_text(json.dumps(result, indent=2) + "\n")
        runner.logger.info("BENCHMARK_RESULT %s", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fp32", "fp32_tuned", "bf16", "bf16_tuned"), required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--videos", type=int, default=160)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    assert args.videos % 4 == 0 and args.videos // 4 > args.warmup
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    assert "L40" in torch.cuda.get_device_name(), "Only L40 is authorized"
    cfg = Config.fromfile("configs/riskprop_nexar.py")
    groups = {"0": [], "1": []}
    with open("data/nexar-collision-prediction/annotations.csv") as source:
        for row in csv.DictReader(source):
            if row["is_test"] == "0" and row["video_id"] not in nexar_val:
                groups[row["target"]].append("train:" + row["video_id"])
    rng = random.Random(args.seed)
    selected = []
    for group in groups.values():
        rng.shuffle(group)
        selected.extend(group[:args.videos // 2])
    cfg.train_dataloader.dataset.video_ids = selected
    cfg.train_dataloader.num_workers = args.workers
    cfg.train_dataloader.persistent_workers = args.workers > 0
    cfg.val_dataloader.dataset.video_ids = None
    cfg.train_cfg.update(max_epochs=1, val_begin=1, val_interval=1)
    cfg.env_cfg.cudnn_benchmark = args.mode.endswith("_tuned")
    cfg.model.backbone.pretrained = None
    cfg.work_dir = args.work_dir
    cfg.launcher = "pytorch"
    cfg.randomness = dict(seed=args.seed, deterministic=False, diff_rank_seed=False)
    cfg.default_hooks.checkpoint = None
    cfg.default_hooks.logger.interval = 10
    cfg.custom_hooks = [dict(type="EpochHook"), dict(
        type="PrecisionBenchmarkHook", warmup=args.warmup, mode=args.mode, seed=args.seed)]
    if args.mode.startswith("bf16"):
        cfg.optim_wrapper.update(type="AmpOptimWrapper", dtype="bfloat16", loss_scale=dict(enabled=False))
    # Full validation remains in FP32 in every arm, isolating training precision.
    cfg.val_cfg.fp16 = False
    if args.skip_validation:
        cfg.val_cfg = cfg.val_dataloader = cfg.val_evaluator = None
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
