# RiskProp training precision: BF16 versus FP32

**Checked:** 2026-11-08  
**Question:** Does the RiskProp paper/upstream config prescribe BF16 or FP32, and can the choice change the trained result?

## Findings

### What the paper and upstream repository specify

The CVPR paper's implementation details report the model, frame size/rate, snippet length, losses, optimizer, epoch count, GPU count, batch size, learning rate, and decay schedule. They do not state FP32, FP16, BF16, AMP, autocast, or a mixed-precision setting. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)

The authors' published snippet config does not declare an AMP wrapper or a reduced-precision dtype. It inherits a plain `optim_wrapper` with SGD from `configs/_base_/schedules/sgd_50e.py`; the public config itself therefore describes the ordinary FP32 training path. It does not prove what precision was used for the paper's reported experiments, because that setting is omitted in the paper and the checked-in config has other differences from the paper. [Published config](https://github.com/xingyueye5/RiskProp/blob/main/configs/predict_anomaly_snippet.py), [SGD schedule](https://github.com/xingyueye5/RiskProp/blob/main/configs/_base_/schedules/sgd_50e.py)

The user's saved full-run config also has a plain SGD `optim_wrapper` without `AmpOptimWrapper` or `dtype`; the corresponding training log reports finite losses/grad norms and peak logger memory 25,203 MiB. That is consistent with ordinary FP32 execution. [Saved run config](../../work_dirs/riskprop_nexar_l40_full/20261104_085904/vis_data/config.py), [training log](../../work_dirs/riskprop_nexar_l40_full/20261104_085904/20261104_085904.log)

### Can BF16 change the learned model and its metrics?

Yes, it can, but the size and direction of any change cannot be determined from the dtype alone. In AMP, eligible operations run at lower precision while other operations stay in FP32; small rounding differences can change gradients and subsequent SGD updates. PyTorch says mixed precision often maintains convergence with little accuracy loss, but numerical behavior and speed depend on the workload, and recommends empirical verification. [PyTorch AMP docs](https://docs.pytorch.org/docs/stable/amp.html), [PyTorch mixed-precision guidance](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)

BF16 has the same exponent range as FP32, so it is less prone to overflow/underflow than FP16, but it has less mantissa precision. It uses 16-bit storage for eligible calculations and can reduce activation memory and improve speed on supported hardware. Neither property guarantees unchanged validation/test metrics. [PyTorch dtype definitions](https://docs.pytorch.org/docs/main/tensor_attributes.html), [PyTorch mixed-precision guidance](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)

MMEngine 0.10.7 exposes this choice through `AmpOptimWrapper(dtype='bfloat16')`; it does not become active merely because the GPU supports BF16. [MMEngine 0.10.7 AmpOptimWrapper](https://mmengine.readthedocs.io/en/v0.10.7/_modules/mmengine/optim/optimizer/amp_optimizer_wrapper.html), [MMEngine mixed-precision setup](https://mmengine.readthedocs.io/en/v0.10.1/common_usage/speed_up_training.html)

The local `tools/analysis_tools/benchmark_nexar_precision.py` contains an experimental BF16 mode that explicitly installs `AmpOptimWrapper(dtype='bfloat16')`. This is a local comparison helper, not an instruction from the paper or upstream release. The search of current `work_dirs` found no saved `result_rank*.json` from that precision comparison, so there is no durable local result here to quote as evidence of metric parity.

## Practical conclusion

- **Paper:** does not prescribe BF16 or FP32.
- **Upstream checked-in code:** uses the ordinary FP32 path; no AMP setting is enabled there.
- **User's recorded full run:** also used the ordinary FP32 path based on its saved config.
- **Effect on outcomes:** BF16 may yield near-identical results, but can shift scores/metrics because it changes arithmetic and the optimization trajectory. There is no basis to promise identical RiskProp performance without a controlled comparison.
- **Comparison needed:** hold data split/order, seed, initialization, global batch/effective optimizer steps, and all hyperparameters fixed; compare finite loss/gradients and validation metrics during a short screening run, then compare complete-run metrics before claiming the final performance is preserved. A short run can catch instability or a large degradation, but cannot prove final metric parity.

## Sources

- [RiskProp CVPR paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)
- [RiskProp authors' public training config](https://github.com/xingyueye5/RiskProp/blob/main/configs/predict_anomaly_snippet.py)
- [MMAction2 SGD schedule](https://github.com/xingyueye5/RiskProp/blob/main/configs/_base_/schedules/sgd_50e.py)
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch guidance on mixed precision](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [MMEngine AmpOptimWrapper 0.10.7](https://mmengine.readthedocs.io/en/v0.10.7/_modules/mmengine/optim/optimizer/amp_optimizer_wrapper.html)
