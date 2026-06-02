# BRICK: Barrier-Aware Indoor Thermal-Field Reconstruction

BRICK is a barrier-aware inductive spatiotemporal kriging model for sparse indoor thermal-field reconstruction and short-horizon forecasting. The code is adapted from KITS and adds indoor connection-aware graph support, wall-aware hidden-state transfer, and a detached residual forecast head.

## Repository layout

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

The experiments were run with Python 3.8, PyTorch 1.8.1, PyTorch Lightning 1.4.0, CUDA 11.1, and one NVIDIA GeForce RTX 4060 Laptop GPU.

Recommended setup:

```bash
conda env create -f env_windows.yaml
conda activate kits
```

A minimal pip-style dependency list is also provided:

```bash
pip install -r requirements.txt
```

## Data

The repository includes the preprocessed Intel Lab data and the semi-synthetic physical-diffusion benchmark used by the paper.

Key statistics:

- Sensors: 42
- Time steps: 1,627 at 30-minute resolution
- Temperature entries: 68,334 total, 56,815 naturally observed
- Window length: 24 time steps
- Forecast horizon: 12 time steps
- Sliding-window samples with forecasting enabled: 1,592
- Train/validation/test windows: 1,115 / 159 / 318, using chronological 70/10/20 splitting
- Room-coverage splits: 25%, 50%, and 75% hidden-sensor rates; each split keeps at least one observed anchor in every room group

Semi-synthetic files are stored as:

```text
data/physical_diffusion/physical_diffusion_offset_{0,5,10,15,20}.npz
```

Each file contains `temperature`, `original_temperature`, `delta_temperature`, `natural_mask`, the diffusion matrix, and metadata. Fixed sensor splits are stored in `data/room_coverage_splits/`.

## Train

Train the main paper configuration on the 50% hidden-sensor, +20 C physical-diffusion setting with the first fixed room-coverage split:

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

Run all three paper seeds by changing `--seed` and the split file suffix to `seed1`, `seed2`, and `seed3`.

## Evaluate

Training automatically evaluates the best checkpoint and writes `predictions.npz` and `forecast_predictions.npz` under `logs/intel_lab/kits/<run_id>/`.

To evaluate an existing checkpoint:

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

The script prints reconstruction MAE/MAPE/MRE/MSE/RMSE/R2 and 12-step forecast MAE/RMSE.

## Regenerate Benchmark Data

The checked-in data should be enough to reproduce the paper settings. To regenerate the semi-synthetic physical-diffusion files, run:

```bash
python scripts/generate_physical_diffusion_semisynthetic_data.py
python scripts/validate_physical_diffusion_data.py
python scripts/generate_room_coverage_splits.py
```

If you regenerate data, verify that the output shapes remain `(1627, 42)` and that each room-coverage split keeps at least one room anchor.

## Citation

If you use this repository, please cite the BRICK paper and the original KITS implementation.
