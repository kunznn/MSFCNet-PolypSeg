# MSFCNet: Multi-Scale Feature Calibration Network for Polyp Segmentation

This repository contains the PyTorch implementation of **MSFCNet**, a multi-scale feature calibration network for colonoscopic polyp segmentation.

MSFCNet uses an ImageNet-pretrained ResNet34 encoder, Multi-Scale Perception Modules (MSMs), and a semantic-guided Feature Calibration Fusion (FCF) decoder. MSM improves the representation of polyps with different scales and irregular appearances, while FCF uses high-level semantic information to calibrate low-level skip features before fusion.

> **Note:** Some files retain legacy names from early experiments. In particular, `train_polypmfc.py` and the `PolypMFCNet` class correspond to the final **MSFCNet** model. In some early experiment files, FCF was temporarily named `BCF`. The checkpoint and configuration named `a3_polypmfc_msm_bcf_*` correspond to the full MSFCNet model reported in the manuscript.

## Repository Structure

```text
configs/          Training configurations
src/              Datasets, models, losses, metrics, and utilities
scripts/          Data preparation, training, evaluation, and profiling scripts
results/          Concise reported tables and the network architecture diagram
logs/             Training log for the full MSFCNet model
checkpoints/      Placeholder for model weights
```

## Environment

Install the required packages with:

```bash
pip install -r requirements.txt
```

The experiments were implemented with PyTorch. CUDA is recommended for training and evaluation.

## Data Preparation

The raw public datasets are not included in this repository. Please download the datasets from their official sources and organize them under `data/raw/`.

After placing the raw datasets, run:

```bash
python scripts/prepare_data.py --seed 42
python scripts/check_dataset.py
```

The processed data will be saved under `data/processed/`.

## Training MSFCNet

The full MSFCNet configuration is:

```text
configs/a3_polypmfc_msm_bcf_kvasir_cvc.yaml
```

Train the model with:

```bash
python scripts/train_polypmfc.py --config configs/a3_polypmfc_msm_bcf_kvasir_cvc.yaml
```

The best checkpoint will be saved to:

```text
checkpoints/a3_polypmfc_msm_bcf_kvasircvc_jointtrain_best.pth
```

## Evaluation

After training, evaluate a checkpoint with:

```bash
python scripts/evaluate.py ^
  --config configs/a3_polypmfc_msm_bcf_kvasir_cvc.yaml ^
  --checkpoint checkpoints/a3_polypmfc_msm_bcf_kvasircvc_jointtrain_best.pth ^
  --output_csv results/tables/a3_polypmfc_msm_bcf_test_results.csv
```

For Linux/macOS shells, replace `^` with `\`.

## Reported Results

The public result files are collected in `results/`:

- `tables/quantitative_results.csv`: Dice comparison on five benchmark datasets.
- `tables/efficiency_results.csv`: model complexity and inference speed comparison.
- `tables/ablation_results.csv`: contributions of MSM and FCF.
- `figures/msfcnet_architecture.png`: flow diagram of the proposed MSFCNet.

The final MSFCNet training log is available at:

```text
logs/a3_polypmfc_msm_bcf_kvasircvc_jointtrain_train_log.csv
```

## Model Weights

Large checkpoint files are not committed to the repository. The final full MSFCNet checkpoint corresponds to:

```text
a3_polypmfc_msm_bcf_kvasircvc_jointtrain_best.pth
```

It can be placed under `checkpoints/` before running the evaluation script.

## Citation

Citation information will be added after the manuscript is available.
