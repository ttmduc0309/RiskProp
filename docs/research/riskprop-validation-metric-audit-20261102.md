# RiskProp validation metric and training protocol audit

Date: 2026-11-02

## Summary

**Historical snapshot:** this audit records the metric names and behavior before the `improve-checkpoint` branch renamed the implementation keys. The paper-protocol discrepancies discussed below remain relevant; only the metric names and selection-key spelling were changed.

The `compare_formula` branch now reports both `mAUC_fpr_0.1_std` (scikit-learn standardized partial AUC) and `mAUC_fpr_0.1_norm` (trapezoidal ROC area through FPR 0.1 divided by 0.1). The latter is a comparison candidate, not a verified paper-exact formula: the cited TOP equation shows raw partial area, while published values exceed the raw area's maximum of 0.1, indicating that a scale/normalization detail is omitted or differs.

The RiskProp paper reports evaluation metrics, but does not document a separate validation split, a validation schedule, or which metric was used to select the final checkpoint. It explicitly says that mTTA is a compatibility metric and is not used for training. The public repository's generic config selects `mAUC@`, but the metric implementation defines `mAUC@` as *unconstrained full AUC*, whereas the paper's primary AUC is constrained at FAR <= 0.1 (`mAUC@0.1` in this code). This is evidence about the released generic config, not evidence that the authors selected the paper's model that way.

The paper reports constrained AUC at FAR <= 0.1, full mAUC, mAP, and constrained mTTA, based on Nexar's evaluation protocol. In the current local evaluator, `AnticipationMetric` computes mAP using unrestricted scikit-learn `average_precision_score`; this appears inconsistent with the paper's statement that AP is computed under FAR <= 0.1. In addition, the local best-checkpoint key `mAUC@` selects full, unconstrained AUC, not the paper's primary constrained AUC. These are concrete candidates for the user's concern that validation metrics do not match the paper. A separate protocol comparison is needed before treating any local metric as paper-equivalent.

## Findings

### What the paper says

- RiskProp uses 50 epochs, 8 NVIDIA A800 GPUs, batch size 64, SGD, initial learning rate 0.002, and decays the LR to 10% of its current value every 20 epochs. It does not state a validation frequency or checkpoint-selection metric. [CVPR 2026 paper, implementation details](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)
- The paper's evaluation section says all metrics use FAR <= 0.1 and describes positive samples from four pre-accident windows. It names constrained AUC and AP, mAUC, and mTTA. It also says mTTA requires anomaly-onset labels and is included for compatibility with prior baselines, not for training. [CVPR 2026 paper, evaluation metrics](https://openaccess.thecvf.com/content/CVPR2026/papers/Zou_RiskProp_Collision-Anchored_Self-Supervised_Risk_Propagation_For_Early_Accident_Anticipation_CVPR_2026_paper.pdf)
- The paper describes training labels and losses, but the phrase “validated the effectiveness” in the dataset paragraph refers to the experiments; the paper does not establish a held-out validation protocol or state that validation chose the checkpoint.

### What the official repository adds

- The public RiskProp GitHub README says the default best-checkpoint criterion is `mAUC@` and describes that key as mean partial AUC under FPR <= 0.1. But the actual metric code creates `mAUC@0.1` for constrained AUC and `mAUC@` for full AUC, so the README's description does not match the code's key names. [Official RiskProp README](https://github.com/xingyueye5/RiskProp), [official metric implementation at the checked-out source commit](https://github.com/xingyueye5/RiskProp/blob/579376fa1d879f28a9d68c2565d1e449de24b666/taa/metrics.py)
- The official generic config runs validation each epoch and selects `mAUC@`, but it activates CAP and leaves Nexar commented out. It therefore documents generic repository behavior, not necessarily the paper's Nexar experiment. [Official generic config at source commit](https://github.com/xingyueye5/RiskProp/blob/579376fa1d879f28a9d68c2565d1e449de24b666/configs/predict_anomaly_snippet.py)
- The local Nexar config validates every 10 epochs and sets `save_best="mAUC@"`, which means it selects unconstrained full mAUC under the current local evaluator. [riskprop_nexar.py](/data/home/ducttm/project/colision_prediction/RiskProp/configs/riskprop_nexar.py:24)
- Repository settings explain what this codebase runs; they do not fill the paper's missing report of which validation metric or split the authors used.

### What the local evaluator currently computes

- The evaluator separates samples marked `is_val` and returns metrics with `@` suffixes for them. [metrics.py](/data/home/ducttm/project/colision_prediction/RiskProp/taa/metrics.py:81)
- It calculates `mAUC@0.1` via `roc_auc_score(..., max_fpr=0.1)`, then averages the three 0.5s, 1.0s, and 1.5s windows. It calculates `mAUC@` separately as the mean of three *full* AUCs. [metrics.py](/data/home/ducttm/project/colision_prediction/RiskProp/taa/metrics.py:172), [full AUC aggregation](/data/home/ducttm/project/colision_prediction/RiskProp/taa/metrics.py:180)
- Scikit-learn documents `max_fpr` as returning a **standardized** partial AUC. The RiskProp paper specifies the FAR-constrained metric but does not provide enough pseudocode in the retrieved section to confirm whether its normalization is identical. Treat local `mAUC@0.1` as a candidate, not a proven exact reproduction. [scikit-learn `roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
- The local Nexar config's `save_best="mAUC@"` therefore selects using full AUC, not constrained AUC. [riskprop_nexar.py](/data/home/ducttm/project/colision_prediction/RiskProp/configs/riskprop_nexar.py:26)
- It calculates `mAP@` by calling `average_precision_score` on the full positive/negative set at each time horizon. There is no FAR <= 0.1 filtering in those AP calls. [metrics.py](/data/home/ducttm/project/colision_prediction/RiskProp/taa/metrics.py:188)
- The paper's wording attaches the FAR <= 0.1 constraint to both AUC and AP. Thus local `mAP@` does not match that description. And `mAUC@0.1` is only a candidate match: negative/positive clip construction, time-window handling, and partial-AUC definition must also align with the Nexar/RiskProp protocol.
- The paper describes an equal number of negative clips from safe-driving segments for the interval evaluation. The local metric code forms a negative clip score for every negative validation video and does not itself equalize the number of negatives to positives. This sampling detail is another item to check against the intended validation protocol.

### Follow-up to the cited TOP evaluation protocol

RiskProp reference [45] is the same authors' earlier NeurIPS 2025 paper, *Accident Anticipation via Temporal Occurrence Prediction*. It gives a precise constrained-AUC formula: trapezoidal area over ROC points up to FAR/FPR lambda, with lambda=0.1 by default; for each anticipation horizon it uses positive clips and an equal number of 0.5-second negative clips from accident-free driving. [TOP paper, §4](https://openreview.net/pdf/cb138a5bdcc90c810d7b6e56e8cff55572d06680.pdf)

The official TOP paper describes a held-out test set of about 20% for CAP/DADA and describes its benchmark evaluation, but it does not document a distinct validation split, validation cadence, or checkpoint selection metric. It is a useful source for the evaluation method RiskProp cites, but it still does not establish how RiskProp selected its final checkpoint.

Because the local implementation calls scikit-learn's `roc_auc_score(max_fpr=0.1)`, which returns a **standardized** partial AUC, while TOP's displayed formula is the unstandardized trapezoidal partial area, the local `mAUC@0.1` is not an exact implementation of the cited formula unless an additional scaling equivalence is demonstrated. [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

## Sources beyond the paper

- The first-party repo README is the only located source that says which metric is the default checkpoint-selection key (`mAUC@`). It gives no validation split description or per-epoch validation protocol beyond the configs.
- The related TOP paper clarifies the constrained-AUC calculation and balanced evaluation sampling referenced by RiskProp, but does not clarify RiskProp's validation/checkpoint selection.
- The lead author's personal publication page links the paper and code, but has no separate training note or blog post with more detailed validation instructions. [Yiyang Zou's page](https://xingyueye5.github.io/)
- Search results surfaced third-party summaries and general Nexar challenge articles, but none provided verifiable additional details about RiskProp's internal validation or checkpoint selection. For the dataset-level submission metric, Nexar's competition page defines mean AP at 500ms, 1000ms, and 1500ms before the event; this is distinct from the paper's expanded constrained-FAR evaluation for its reported comparisons. [Nexar Kaggle evaluation](https://www.kaggle.com/competitions/nexar-collision-prediction)

## Assessment

The best-supported answer is: the paper reports the benchmark metrics, but does not say which one it used for validation/checkpoint selection. The released generic config uses `mAUC@` as its selection key, but in code that means full, unconstrained AUC; the local Nexar config inherits the same key. The local evaluator's `mAP@` also does not appear to honor the paper's stated FAR constraint. Neither should be treated as paper-faithful without correction and an end-to-end protocol audit.

## Next verification steps

1. Compare local positive/negative clip sampling (including equal negative sampling) and temporal-window construction against the paper and Nexar protocol.
2. Define constrained AP precisely from the cited protocol and add a separate metric key instead of silently changing reported `mAP`.
3. Evaluate each validation checkpoint with both existing and paper-matched metrics on the same fixed split, then compare checkpoint ranking. Keep the official held-out test data out of this selection process.
# Metric naming update (2026-11-10)

This is a historical audit of the earlier implementation, which emitted both
standardized and normalized partial AUC. Current RiskProp code emits only the
normalized metric as `mAUC_fpr_0.1@` during validation and `mAUC_fpr_0.1` in the
Nexar test evaluator. References below to old metric names describe that
earlier code and its results, not current output.
