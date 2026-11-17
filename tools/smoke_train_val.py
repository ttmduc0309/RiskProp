#!/usr/bin/env python3
"""Run a short, balanced training/validation smoke from an existing config.

Example (select only GPUs you are allowed to use):
  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone \
    --nproc_per_node=2 tools/smoke_train_val.py configs/predict_anomaly_frame.py

Run from the repository root in the RiskProp environment. This uses the dataset
and preprocessing defined by the config; it does not convert data or fix paths.
"""
import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mmengine.config import Config, DictAction
from mmengine.dist import is_main_process
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmaction.registry import DATASETS
from mmaction.utils import register_all_modules
from mmengine.utils import import_modules_from_strings


def balanced_subset(dataset, limit, validation=False):
    """Choose equal numbers of positive/negative samples, excluding test."""
    groups = {0: [], 1: []}
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        label = info.get('target')
        if info.get('is_test', False) or label is None:
            continue
        if validation and not info.get('is_val', False):
            continue
        groups[int(bool(label))].append(index)
    count = min(limit // 2, len(groups[0]), len(groups[1]))
    if count < 1:
        raise ValueError('Smoke needs at least one positive and one negative sample; '
                         f'found negative={len(groups[0])}, positive={len(groups[1])}.')
    indices = [i for pair in zip(groups[0][:count], groups[1][:count]) for i in pair]
    return dataset.get_subset(indices), len(groups[0]) + len(groups[1])


class SmokeTiming(Hook):
    priority = 'LOWEST'

    def __init__(self, full_train, full_val, epochs, gpu_count):
        self.full_train, self.full_val = full_train, full_val
        self.epochs, self.gpu_count = epochs, gpu_count
        self.train_seconds = None

    def before_train_epoch(self, runner):
        self.train_start = time.perf_counter()

    def after_train_epoch(self, runner):
        self.train_seconds = time.perf_counter() - self.train_start

    def before_val(self, runner):
        self.val_start = time.perf_counter()

    def after_val(self, runner):
        val_seconds = time.perf_counter() - self.val_start
        if not is_main_process():
            return
        train_batches = len(runner.train_dataloader)
        val_batches = len(runner.val_dataloader)
        train_batch = runner.train_dataloader.batch_size
        val_batch = runner.val_dataloader.batch_size
        full_train_batches = math.ceil(self.full_train / self.gpu_count / train_batch)
        full_val_batches = math.ceil(self.full_val / self.gpu_count / val_batch)
        train_est = self.train_seconds / train_batches * full_train_batches
        val_est = val_seconds / val_batches * full_val_batches
        print('\nSMOKE TIMING (wall time; includes startup effects)', flush=True)
        print(f'train: {self.train_seconds:.2f}s / {train_batches} batches per rank; '
              f'full epoch estimate {train_est / 60:.2f} min', flush=True)
        print(f'validation: {val_seconds:.2f}s / {val_batches} batches per rank; '
              f'full validation estimate {val_est / 60:.2f} min', flush=True)
        print(f'full train + validation estimate for {self.epochs} epochs: '
              f'{(train_est + val_est) * self.epochs / 3600:.2f} hours', flush=True)
        print('Estimate assumes validation each epoch; checkpoint writes, later-epoch '
              'variation and contention are not measured. Tiny samples are noisy.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--train-samples', type=int, default=64)
    parser.add_argument('--val-samples', type=int, default=32)
    parser.add_argument('--work-dir', default='work_dirs/smoke_train_val')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    args = parser.parse_args()
    if min(args.train_samples, args.val_samples) < 2:
        parser.error('Both sample limits must be at least 2.')
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules(init_default_scope=True)
    if cfg.get('custom_imports'):
        import_modules_from_strings(**cfg.custom_imports)
    full_epochs = cfg.train_cfg.max_epochs
    gpu_count = int(os.environ.get('WORLD_SIZE', '1')) if args.launcher != 'none' else 1
    if args.launcher == 'pytorch' and 'RANK' not in os.environ:
        parser.error('Use torch.distributed.run (see --help), or --launcher none.')
    full_sizes = []
    for key, limit, validation in (('train_dataloader', args.train_samples, False),
                                   ('val_dataloader', args.val_samples, True)):
        dataset = DATASETS.build(copy.deepcopy(cfg[key].dataset))
        subset, full_size = balanced_subset(dataset, limit, validation)
        cfg[key].dataset = subset
        full_sizes.append(full_size)
        if int(os.environ.get('RANK', '0')) == 0:
            print(f'{key}: smoke={len(subset)}, eligible full={full_size}', flush=True)
    cfg.train_cfg.max_epochs = 1
    cfg.train_cfg.val_begin = 1
    cfg.train_cfg.val_interval = 1
    cfg.work_dir = args.work_dir
    cfg.launcher = args.launcher
    cfg.default_hooks.checkpoint = None
    cfg.resume = False
    runner = Runner.from_cfg(cfg)
    runner.register_hook(SmokeTiming(*full_sizes, full_epochs, gpu_count))
    runner.train()


if __name__ == '__main__':
    main()
