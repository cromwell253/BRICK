# BRICK

BRICK reconstructs indoor temperatures from sparse sensor observations using a geometry-conditioned residual representation and observation-conditioned query inference.

## Repository contents

| Location | Contents |
|---|---|
| `brick/model.py` | BRICK model |
| `brick/data.py` | Data loading and query masking |
| `brick/training.py` | Training objectives and sampling |
| `brick/inference.py` | Query prediction |
| `data/` | Building temperature, coordinate and connection tables |
| `configs/` | Experiment settings, data partitions, normalization and sample plans |
| `checkpoints/` | 54 frozen comparison checkpoints |
| `results/` | Comparison and frozen-transfer metrics |
| `figures/` | Whole-floor reconstruction comparisons |

## Datasets

The eight published evaluation datasets cover hospital, office and residential layouts. Each file contains 1009 frames at 10-minute intervals, spanning seven days. Temperatures are in degrees Celsius and coordinates are in metres. Hospital_2, Office_1, Residence_2, Office_2, Residence_1 and Residence_3 are used for comparison and ablation. Hospital_1 and Residence_4 are used for frozen cross-building evaluation.

| Dataset | Temperature series | Plan dimensions X × Y (m) | Evaluation dates |
|---|---:|---|---|
| Hospital_1 | 46 | 20.00 × 12.00 | 2026-07-06 to 2026-07-13 |
| Hospital_2 | 60 | 18.75 × 11.89 | 2025-12-08 to 2025-12-15 |
| Office_1 | 27 | 17.25 × 8.05 | 2026-07-13 to 2026-07-20 |
| Office_2 | 41 | 14.90 × 8.23 | 2025-12-01 to 2025-12-08 |
| Residence_1 | 16 | 12.50 × 9.40 | 2026-06-22 to 2026-06-29 |
| Residence_2 | 28 | 13.16 × 9.37 | 2025-12-15 to 2025-12-22 |
| Residence_3 | 39 | 20.00 × 12.00 | 2026-07-06 to 2026-07-13 |
| Residence_4 | 45 | 21.58 × 14.56 | 2026-01-05 to 2026-01-12 |

Each dataset contains three tables.

| File | Fields |
|---|---|
| `coordinates.csv` | Node and sensor identifiers, x/y coordinates, public connected-component identifiers |
| `temperature.csv` | Frame, timestamp and one temperature column per node |
| `connection_probability.csv` | Source/target node identifiers, connection probability and intervening wall count |

Dataset versions are recorded in `data/datasets.json`.

## Connection information

The building pair tables use a geometry-dependent connection prior based on distance and intervening wall count. BRICK normalizes the probability matrix P by its maximum and incorporates it into graph edge weights through A = A_distance × (0.5 + 0.5P). Both graph thresholds are 0.1, with zero diagonal entries. The supplied [54-sensor analysis](docs/connection_analysis.md) illustrates the association between connection probability, distance and wall separation.

## Running the code

Install the dependencies with Python 3.10 or later.

```sh
pip install -r requirements.txt
python -m brick predict --scene Hospital_2 --fold 0 --seed 505 --output outputs/hospital2
python -m brick score --scene Hospital_2 --fold 0 --seed 505 --output outputs/hospital2
python -m brick train --scene Hospital_2 --fold 0 --seed 505 --device cuda --output outputs/training
```

## Experimental configuration

Comparison uses six datasets, three spatial folds and seeds 505, 606 and 707. Ablation uses the same datasets and folds with seeds 1103, 1201 and 1301, covering the baseline, module A, module B and A+B. Each learned model has 54 conditions. The chronological split uses frames 0–707 for training, 708–807 for development and 808–1008 for the temporal holdout. Query histories are masked throughout the 24-step input window. Training uses 48 epochs and 384 updates. BRICK uses 128 samples per epoch, batch size 16, Adam with learning rate 0.001, hidden width 64 and gradient clipping at 5. Development evaluation occurs every two epochs. MAE and RMSE are averaged over folds and seeds within each dataset, followed by an equal average over the six datasets. Frozen-transfer results also include R². The recorded environment uses PyTorch 2.3.0, CUDA 11.8 and an NVIDIA GeForce RTX 4060 Laptop GPU.

Detailed settings, kernel parameters and result tables are available in [experimental details](docs/experiment_details.md) and the `configs/` and `results/` directories.

## License

The KITS backbone attribution and S-Lab License 1.0 are provided in `NOTICE` and `LICENSE`.
