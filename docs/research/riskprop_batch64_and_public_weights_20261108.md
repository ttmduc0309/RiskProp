# RiskProp batch 64 feasibility and public weights

**Checked:** 2026-11-08  
**Scope:** Clarify the paper's batch-size statement, compare it with the published training config and the local Nexar run, and check whether the authors publish a trained RiskProp checkpoint.

## Findings

### What the paper reports

The CVPR paper says training ran for 50 epochs on 8 NVIDIA A800 GPUs with batch size 64, initial learning rate 0.002, and a 10x decay every 20 epochs. The paper does not define whether 64 is the global batch or the per-GPU batch, and does not state the A800 memory SKU, precision, or peak VRAM use. [CVF paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)

### What the public repository reports

The paper links to the authors' [RiskProp GitHub repository](https://github.com/xingyueye5/RiskProp). Its README gives an 8-GPU training command, while the released `predict_anomaly_snippet.py` config sets `train_dataloader.batch_size=2`, `clip_len=5`, and `num_clips=30`. In the repository's MMEngine distributed setup, dataloader batch size is per process/GPU; thus this checked-in config corresponds to global batch 16 when launched on 8 GPUs. That differs from the paper's stated 64, so the public config alone does not establish what exact config produced the paper results. [README](https://github.com/xingyueye5/RiskProp#training), [published config](https://github.com/xingyueye5/RiskProp/blob/main/configs/predict_anomaly_snippet.py)

The local Nexar training log likewise records batch size 2 and `memory: 25203` (MiB); its config uses 5 frames per clip, 30 clips, and 224×224 frames. [Local config](../../work_dirs/riskprop_nexar_l40_full/20261104_085904/vis_data/config.py), [local log](../../work_dirs/riskprop_nexar_l40_full/20261104_085904/20261104_085904.log)

### Memory estimate for 8 per GPU

If paper batch 64 means global batch 64 over 8 GPUs, that is 8 samples/GPU. Starting from the local observed 25,203 MiB at 2 samples/GPU, a simple linear extrapolation gives:

```text
25,203 MiB × (8 / 2) = 100,812 MiB ≈ 98.5 GiB per GPU
```

This is a rough estimate, not a measured requirement: model/optimizer/context memory is fixed, while activations dominate and scale with batch; memory allocation and kernels can also make scaling nonlinear. Correcting for fixed overhead could lower the estimate, but the local measurement still makes 8/GPU look near or over the capacity of an 80-GB A800 in FP32. The estimate therefore does not prove the authors could not run it: their precision, exact config, activation handling, and GPU SKU are not specified. If 64 instead meant per GPU, the local linear estimate would be roughly 788 GiB/GPU, making that interpretation implausible for ordinary A800 cards. These are inferences from the local run and paper, not a claim about the authors' undisclosed setup.

### Are trained RiskProp weights public?

The checked official repository has no GitHub releases, no RiskProp checkpoint in its tracked files, and its README tells users to load a checkpoint from `work_dirs/` after training. It does provide a pretrained SlowOnly-R50/Kinetics710 checkpoint URL, which is a backbone initialization, not the trained RiskProp model. [Repository releases](https://github.com/xingyueye5/RiskProp/releases), [README](https://github.com/xingyueye5/RiskProp), [upstream SlowOnly checkpoint](https://download.openmmlab.com/mmaction/v1.0/recognition/slowonly/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb/slowonly_imagenet-pretrained-r50_32xb8-8x8x1-steplr-150e_kinetics710-rgb_20230612-12ce977c.pth)

The first-party CVF/arXiv record and the first author's publication page link the paper and code, but the material located in this check does not expose a supplement with more training settings or a RiskProp-trained checkpoint. [CVF record](https://openaccess.thecvf.com/content/CVPR2026/html/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.html), [author page](https://xingyueye5.github.io/)

## Conclusion

- The public evidence does **not** establish that 64 is per GPU; global batch 64 (8/GPU) is the more plausible interpretation, but remains an inference.
- The published upstream config uses 2 samples/GPU, or global batch 16 on 8 GPUs. It does not match the paper's reported batch 64, so the paper's exact recipe may have used an unpublished override or a different config.
- Extrapolating the user's FP32 run suggests 8/GPU may exceed one 80-GB GPU. A matching hardware/config smoke test would be needed to confirm; do not treat the linear estimate as a guaranteed OOM result.
- No official RiskProp-trained checkpoint was found in the checked author repository/releases. Only the pretrained backbone initialization is directly downloadable there.

## Sources

- [CVPR 2026 paper (CVF)](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)
- [Authors' repository](https://github.com/xingyueye5/RiskProp)
- [Repository training config](https://github.com/xingyueye5/RiskProp/blob/main/configs/predict_anomaly_snippet.py)
- [Repository releases](https://github.com/xingyueye5/RiskProp/releases)
- [First author's publication page](https://xingyueye5.github.io/)
