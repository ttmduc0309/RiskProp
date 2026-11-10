# Nexar frame sampling and metric inputs across repositories

Reviewed: 2026-11-10. Scope: public TOP, RiskProp, PRE-ACT, BADAS-Open, and the
Nexar evaluator shipped with this workspace. The comparison distinguishes
frame/window construction from the later selection of scores used by metrics.

## Findings

| Implementation | What frames/windows feed evaluation? | Does label affect sampling? | What the metric consumes |
|---|---|---|---|
| Nexar submission evaluator | It does not decode videos or select frames. It accepts one submitted score per video. | No frame sampling in evaluator. | Official mAP is AP per Nexar group, averaged across groups; this evaluator does not calculate mAUC/pAUC. [Workspace evaluator](../../../datasets/Nexar/evaluate_submission.py#L7-L26) |
| TOP public Nexar config | `SampleFramesBeforeAccident` runs in validation/test mode. Positive windows are aligned to the accident frame; negative windows begin at the video start when possible, otherwise are aligned about 3 seconds before an accident frame. | **Yes.** The transform branches on positive/negative target. | TOP builds positive scores at the 0.5/1.0/1.5-second horizons and a negative score from the last five predictions in each sampled sequence. Both mAP and mAUC use these label-selected windows. The Nexar config points `test_dataloader` to `val_dataloader`, so this released setup evaluates its annotated validation split, not the hidden competition test. [Nexar config](https://github.com/happytianhao/TOP/blob/main/configs/accident_anticipation/top/top_nexar.py#L15-L32), [validation pipeline](https://github.com/happytianhao/TOP/blob/main/configs/accident_anticipation/top/top_cap.py#L45-L54), [sampler](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/transforms.py#L45-L77), [metrics](https://github.com/happytianhao/TOP/blob/main/accident_anticipation/metrics.py#L180-L208) |
| RiskProp public `main` | Uses `SampleFramesBeforeAccident`. With known labels, positives end at the annotated accident frame; negatives are sampled from the beginning if possible, otherwise from roughly 3 seconds before the accident. | **Yes** when labels/event frame are available. With `target=None`, the public hidden-test path samples the video tail for every video. [Sampler](https://github.com/xingyueye5/RiskProp/blob/main/taa/transforms.py#L54-L90) |
| User's RiskProp local `dev` | Validation still uses the label-aware path above. Its submission-test branch (`is_test=True`) explicitly samples the observed tail identically for every video; full-timeline TTA uses a separate transform. | Validation: **yes**. Submission benchmark: **no**, due to the local label-blind tail branch. | Validation mAP/mAUC aggregate horizon scores from the validation predictions. Test mAP uses predictions generated from the common tail windows; labels are joined after inference in the benchmark evaluator. [Local sampler](../../taa/transforms.py#L47-L91), [Nexar config](../../configs/riskprop_nexar.py#L11-L29), [benchmark evaluator](../../tools/analysis_tools/evaluate_nexar_submission.py#L220-L260) |
| PRE-ACT public Nexar validation | Default `eval_mode` is `sliding_window`; the Nexar training script also passes that mode explicitly. With the released Nexar metadata, it slides over the observed clip at a common temporal stride for positives and negatives. It assigns per-window targets after constructing the windows. | **No** for the default Nexar `sliding_window` mode. `binary_clips` is an alternate mode that does choose positive and negative clips differently, so that mode is label-conditioned. [Default mode](https://github.com/giddyyupp/PRE-ACT/blob/master/train_autoencoder_anticipation_ddp.py#L160-L164), [Nexar launch script](https://github.com/giddyyupp/PRE-ACT/blob/master/slurm_scripts/slurm_videomae_Nexar.sh#L49-L53), [sliding-window construction](https://github.com/giddyyupp/PRE-ACT/blob/master/dataloaders/dataloader_nexar.py#L611-L705), [alternate binary mode](https://github.com/giddyyupp/PRE-ACT/blob/master/dataloaders/dataloader_nexar.py#L712-L785) |
| BADAS-Open public inference | The public wrapper builds 16-frame inputs at 8 FPS and calls a sliding-window model to produce a sequence of window scores. Inference receives the video, not its Nexar label. | **No** in the inference API. | The public repo exposes risk predictions, but does not provide a Nexar mAP/mAUC evaluation script. Its README's benchmark table alone does not specify a metric-ready frame-to-score protocol. [Loader](https://github.com/getnexar/BADAS-Open/blob/main/badas/badas_loader.py#L41-L49), [prediction API](https://github.com/getnexar/BADAS-Open/blob/main/badas/badas_loader.py#L79-L94), [preprocessing](https://github.com/getnexar/BADAS-Open/blob/main/badas/badas_loader.py#L120-L175) |

## Practical comparison

- **Label-conditioned:** TOP and RiskProp validation choose different temporal regions for positive and negative videos. Their metric calculations then operate on scores from those different regions. That can make a validation metric look easier than a label-blind submission benchmark because the ground-truth class influenced the model input window.
- **Label-blind sliding/full-timeline:** PRE-ACT's default Nexar validation scans the clip and assigns window labels afterward. RiskProp local `dev` follows a label-blind common-tail rule for its Nexar submission benchmark, but its validation sampler remains label-conditioned.
- **No model-side frame choice:** The Nexar evaluator receives a video-level score. It cannot prescribe which frames a model used to make that score. The submitter owns that choice.
- **BADAS limitation:** BADAS-Open's inference API is label-blind and temporal, but its public repository does not establish an official Nexar mAP/mAUC evaluator. Do not infer benchmark equivalence from the README's AP/AUC table alone.

## Conclusion

There is no single universal Nexar frame-sampling protocol across these repos.
TOP and the public RiskProp anticipation validation path are label-conditioned;
PRE-ACT's default Nexar validation is sliding-window and label-blind; the local
RiskProp submission benchmark uses a common tail; and Nexar's official evaluator
only scores the video-level values it receives. Comparisons should therefore
record the score-window policy alongside mAP/mAUC, not just the metric formula.
