# Running RiskProp on Nexar

This checkout was prepared from upstream commit `579376fa1d879f28a9d68c2565d1e449de24b666` and tested in the Conda environment `riskprop`.

## Environment

The working versions on this server are Python 3.8.5, PyTorch 2.1.2+cu121, torchvision 0.16.2+cu121, MMEngine 0.10.7, MMCV 2.1.0, MMAction2 1.2.0, Decord 0.6.0, scikit-learn 1.3.2, OpenCV contrib 4.10.0.84, and einops 0.8.0. MMCV 2.1.0 is used because this checkout's `requirements/mminstall.txt` requires `<2.2.0`; the old README's 2.2.0 instruction conflicts with that constraint.

To recreate the environment:

```bash
conda create -y -n riskprop -c conda-forge --override-channels python=3.8.5 pip
conda run -n riskprop python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
conda run -n riskprop python -m pip install openmim
conda run -n riskprop mim install mmengine==0.10.7
conda run -n riskprop mim install mmcv==2.1.0
conda run -n riskprop python -m pip install --no-deps -e .
conda run -n riskprop python -m pip install decord==0.6.0 scikit-learn==1.3.2 opencv-python==4.10.0.84 opencv-contrib-python==4.10.0.84 einops==0.8.0
```

The commands use `conda-forge` without accepting or changing Anaconda channel terms. Run the distributed scripts from this Conda environment so they use its `torchrun` and Python.

## Nexar preparation

The converter uses Nexar's existing MP4 files directly. It creates `annotations.csv` and relative symbolic links under `data/nexar-collision-prediction/{train,test}`; it does not extract or recompress the videos. Each annotation stores the video's decoded frame count, actual average FPS, the alert frame, and the event frame. Sampling converts that source frame rate to approximately 10 samples per second, including for variable rates such as 30.5 FPS.

```bash
conda run -n riskprop python tools/prepare_nexar.py \
  --source-root /data/home/ducttm/project/colision_prediction/datasets/Nexar \
  --output-root data/nexar-collision-prediction
```

The existing conversion produced 1,500 training links and 1,344 test links. Because the links point to the existing dataset, the prepared directory occupies about 168 KB beyond the existing 30 GiB media. Do not move only the RiskProp directory without also preserving the linked Nexar source path, or recreate the links with the converter after moving it.

## Smoke training and evaluation

`configs/riskprop_nexar_smoke.py` selects four balanced training examples, 40 validation examples, and two test examples. It trains one epoch and writes the two test predictions to `work_dirs/riskprop_nexar_smoke/test_predictions.csv`. This checks data loading, model forward/backward, validation metrics, checkpoint writing, and prediction export. Its scores are only a pipeline check.

`configs/riskprop_nexar.py` uses all 1,200 train IDs after the repository's 300-ID validation split, evaluates validation without reading official test videos each epoch, and provides a separate test loader for exporting official test predictions. It runs 50 epochs with batch size two per GPU; on two L40s that is global batch four, so it is a full-data operational run but not the paper's batch-64 setup.

```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=29516 conda run -n riskprop \
  bash tools/dist_train.sh configs/riskprop_nexar_smoke.py 2 \
  --work-dir work_dirs/riskprop_nexar_smoke_l40 \
  --cfg-options train_dataloader.batch_size=2
```

For ordinary evaluation after training:

```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=29517 conda run -n riskprop \
  bash tools/dist_test.sh configs/riskprop_nexar_smoke.py \
  work_dirs/riskprop_nexar_smoke_l40/epoch_1.pth 2
```

Check that the selected GPU is available before running. The full-data config points at this prepared dataset and uses the repository's `nexar_val` split.

The full-data config is ready to launch with:

```bash
CUDA_VISIBLE_DEVICES=0,1 PORT=29518 conda run -n riskprop \
  bash tools/dist_train.sh configs/riskprop_nexar.py 2 \
  --work-dir work_dirs/riskprop_nexar_full
```

## Scope and reproduction limitations

This setup establishes a working Nexar training/evaluation path. It does not claim reproduction of the paper. The public loss still differs from the paper's positive-only FFR/confidence handling, the collision BCE weight is unspecified in the paper, and the published Nexar partial-AUC value is below the lower bound of the current scikit-learn standardized partial AUC implementation. Confirm the authors' evaluator/protocol and settle these loss details before using a full run as a paper reproduction. See [`../docs/research/riskprop-reproducibility-audit.md`](../../docs/research/riskprop-reproducibility-audit.md).
