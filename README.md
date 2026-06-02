# BRICK: Barrier-Aware Indoor Thermal-Field Reconstruction

BRICK is a barrier-aware inductive spatiotemporal kriging framework for sparse indoor thermal-field reconstruction and short-horizon forecasting. It extends a KITS-style inductive kriging backbone with indoor connection-aware graph supports, wall-aware hidden-state transfer correction, and a detached residual forecast head.

This repository contains the code, processed Intel Lab data, physically constrained semi-synthetic thermal-diffusion benchmark, fixed room-coverage sensor splits, and commands needed to reproduce the BRICK experiments reported in the paper.

## Repository Layout

```text
configs/                     Final BRICK/KITS configuration files
lib/                         Model, dataset, datamodule, metrics, and utilities
data/intel_lab/              Preprocessed Intel Berkeley Lab sensor data and indoor matrices
data/physical_diffusion/     Physically constrained semi-synthetic thermal-diffusion data
data/room_coverage_splits/   Fixed room-coverage sensor splits used in the paper
scripts/                     Data-generation and validation helpers
train.py                     Training and evaluation entry point
```

## Environment

The reported experiments used:

- Python 3.8
- PyTorch 1.8.1
- PyTorch Lightning 1.4.0
- CUDA 11.1
- NVIDIA GeForce RTX 4060 Laptop GPU

Recommended Conda setup:

```bash
conda env create -f env_windows.yaml
conda activate kits
```

A minimal pip-style dependency list is also provided for reference:

```bash
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build matching your local CUDA version if CUDA 11.1 is not available.

## Data

The repository includes the preprocessed Intel Lab data and the semi-synthetic physical-diffusion benchmark used by BRICK.

Key statistics:

| Item | Value |
| --- | ---: |
| Sensors | 42 |
| Time steps | 1,627 |
| Sampling interval | 30 minutes |
| Temperature entries | 68,334 |
| Naturally observed temperature entries | 56,815 |
| Input window | 24 time steps |
| Forecast horizon | 12 time steps |
| Forecast-enabled sliding-window samples | 1,592 |
| Train / validation / test windows | 1,115 / 159 / 318 |

Semi-synthetic physical-diffusion files are stored as:

```text
data/physical_diffusion/physical_diffusion_offset_{0,5,10,15,20}.npz
```

Each file contains `temperature`, `original_temperature`, `delta_temperature`, `natural_mask`, `source_profile`, indoor coordinates, wall matrix, diffusion weights, and metadata. Fixed room-coverage sensor splits are stored in `data/room_coverage_splits/`. The 25%, 50%, and 75% hidden-sensor settings hide 10, 21, and 31 sensors, respectively, while retaining at least one observed anchor in every room group.

## Quickstart

Run the main BRICK configuration on the 50% hidden-sensor, +20 C physical-diffusion setting with seed 1:

```bash
python train.py \
  --config configs/brick_residual_forecast_detached_d64.yaml \
  --dataset-name intel_lab \
  --model-name kits \
  --seed 1 \
  --miss-rate 0.5 \
  --semisynth-file data/physical_diffusion/physical_diffusion_offset_20.npz \
  --split-file data/room_coverage_splits/split_roomcov_mr0.5_seed1.npz \
  --checkpoint-save-top-k 1
```

The script trains the model, evaluates reconstruction on the held-out sensors, evaluates the 12-step forecast head, and writes predictions under `logs/intel_lab/kits/<run_id>/`.

## Training Commands

Main BRICK configuration used in the paper:

```bash
python train.py \
  --config configs/brick_residual_forecast_detached_d64.yaml \
  --dataset-name intel_lab \
  --model-name kits \
  --seed 1 \
  --miss-rate 0.5 \
  --semisynth-file data/physical_diffusion/physical_diffusion_offset_20.npz \
  --split-file data/room_coverage_splits/split_roomcov_mr0.5_seed1.npz \
  --checkpoint-save-top-k 1
```

KITS-style baseline using the no-structure configuration:

```bash
python train.py \
  --config configs/brick_rf_d_info_none_d64.yaml \
  --dataset-name intel_lab \
  --model-name kits \
  --seed 1 \
  --miss-rate 0.5 \
  --semisynth-file data/physical_diffusion/physical_diffusion_offset_20.npz \
  --split-file data/room_coverage_splits/split_roomcov_mr0.5_seed1.npz \
  --checkpoint-save-top-k 1
```

Connection-only ablation:

```bash
python train.py \
  --config configs/brick_rf_d_conn_d64.yaml \
  --dataset-name intel_lab \
  --model-name kits \
  --seed 1 \
  --miss-rate 0.5 \
  --semisynth-file data/physical_diffusion/physical_diffusion_offset_20.npz \
  --split-file data/room_coverage_splits/split_roomcov_mr0.5_seed1.npz \
  --checkpoint-save-top-k 1
```

To reproduce the three-seed protocol, change both `--seed` and `--split-file` to matching suffixes: `seed1`, `seed2`, and `seed3`.

## Evaluation Commands

Training automatically evaluates the best checkpoint. To evaluate an existing checkpoint, pass `--pretrained-model`:

```bash
python train.py \
  --config configs/brick_residual_forecast_detached_d64.yaml \
  --dataset-name intel_lab \
  --model-name kits \
  --seed 1 \
  --miss-rate 0.5 \
  --semisynth-file data/physical_diffusion/physical_diffusion_offset_20.npz \
  --split-file data/room_coverage_splits/split_roomcov_mr0.5_seed1.npz \
  --pretrained-model logs/intel_lab/kits/<run_id>/best.ckpt
```

The script prints reconstruction MAE/MAPE/MRE/MSE/RMSE/R2 and 12-step forecast MAE/RMSE. It also saves:

```text
logs/intel_lab/kits/<run_id>/predictions.npz
logs/intel_lab/kits/<run_id>/forecast_predictions.npz
```

## Reproducing Paper Settings

Use the following axes to reproduce the main paper settings:

- Hidden-sensor rates: `0.25`, `0.5`, `0.75`
- Physical-diffusion offsets: `0`, `5`, `10`, `15`, `20`
- Seeds: `1`, `2`, `3`
- Main config: `configs/brick_residual_forecast_detached_d64.yaml`

Example for offset 10 and 75% hidden sensors:

```bash
python train.py \
  --config configs/brick_residual_forecast_detached_d64.yaml \
  --dataset-name intel_lab \
  --model-name kits \
  --seed 2 \
  --miss-rate 0.75 \
  --semisynth-file data/physical_diffusion/physical_diffusion_offset_10.npz \
  --split-file data/room_coverage_splits/split_roomcov_mr0.75_seed2.npz \
  --checkpoint-save-top-k 1
```

## Regenerating Benchmark Data

The checked-in data are sufficient for the reported experiments. To regenerate benchmark files, run:

```bash
python scripts/generate_physical_diffusion_semisynthetic_data.py
python scripts/validate_physical_diffusion_data.py
python scripts/generate_room_coverage_splits.py
```

After regeneration, verify that physical-diffusion arrays remain `(1627, 42)` and that every room-coverage split keeps at least one observed room anchor.

## Troubleshooting

If Git or Python runs through an invalid local proxy, clear proxy variables for the current shell:

```bash
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY GIT_HTTP_PROXY GIT_HTTPS_PROXY
```

On PowerShell:

```powershell
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""
$env:GIT_HTTP_PROXY=""
$env:GIT_HTTPS_PROXY=""
```

If CUDA is unavailable, the training script falls back to CPU, but runtime will be slower.

## Citation

If you use this repository, please cite the BRICK paper and the original KITS implementation.
