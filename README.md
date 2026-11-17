# BRICK

Barrier-aware Residual Inference with Connectivity-conditioned Kriging for indoor temperature reconstruction.

BRICK combines a geometry-conditioned residual representation with observation-conditioned query inference. The first module uses component-conditioned ordinary kriging, leave-one-donor-out residuals and an observation-dependent RMS scale. The second module combines the KITS graph backbone with a local query readout and an independently supervised correction for queries without same-component observations.

## Installation

Python 3.10 or later is required. The recorded experiments used PyTorch 2.3.0 with CUDA 11.8 and an NVIDIA GeForce RTX 4060 Laptop GPU. CPU inference is supported.

```sh
pip install -r requirements.txt
python -m brick report
python -m brick predict --scene Hospital_2 --fold 0 --seed 505 --output outputs/hospital2
python -m brick score --scene Hospital_2 --fold 0 --seed 505 --output outputs/hospital2
```

Prediction writes a sealed array before the separate scoring command reads development query endpoints. The default checkpoint is selected by scene, fold and seed. The release includes 54 unchanged final-model checkpoints for the six-scene comparison, with file hashes in `checkpoints/manifest.json`.

## Data

The release contains the eight processed building temperature datasets supplied for this study. Their retained scenario metadata identifies dense-PDE generated temperature sequences. Hospital_2 and the other v7 scenes use `dense-pde-v7-wall-mounted-device-projected-plume`. Office_1, Residence_3 and Residence_4 use the corresponding v10, v12 and v11 processed versions. Exact version strings are recorded in `data/datasets.json`. The processing metadata supports describing these files as simulated building-temperature data.

Each scene contains three tables.

- `coordinates.csv` contains zero-based node identifiers, source sensor identifiers, coordinates in metres and public connected-component identifiers. Components were computed using four-neighbour connectivity over valid non-wall geometry and nearest-grid mapping. Component identifiers are geometry-derived metadata.
- `temperature.csv` contains a frame index, timestamp and one temperature column per node. Temperatures are in degrees Celsius. All eight sequences contain 1009 frames at 10-minute spacing, spanning seven days.
- `connection_probability.csv` contains ordered node pairs, their connection probabilities and wall counts. The node order is identical across all three tables. The geometry quantities enter the fixed graph construction and component kernel. Sensor descriptions, site images, dense grids, event logs, GIFs and duplicate temperature tables have been removed from this publication copy.

| Dataset | Nodes | Role |
|---|---|---|
| Hospital_2 | 60 | Development comparison |
| Office_1 | 27 | Development comparison |
| Residence_2 | 28 | Development comparison |
| Office_2 | 41 | Development comparison |
| Residence_1 | 16 | Development comparison |
| Residence_3 | 39 | Development comparison |
| Hospital_1 | 46 | Frozen cross-building evaluation |
| Residence_4 | 45 | Frozen cross-building evaluation |

The published temperature export preserves the original table values. Float32 training observations were verified exactly against the evaluated six-scene bundles for each approved training partition. Data publication is separate from the original training permissions. The supplied training command supports the six development scenes and their registered partitions.

## Experimental protocol

The 18 building-fold partitions and kernel settings are in `configs/splits.json`. Approximately 40% of nodes were reserved in the outer split. The remaining nodes were partitioned into three disjoint spatial folds. Each fold uses its `train_nodes` as Ki and its `development_query_nodes` as cold queries. The complete query history is masked. Each prediction uses the distance-ordered nearest `len(Ki)-1` eligible donors, excluding the training query itself.

Frames 0–707 form the training interval. Frames 708–807 form the development interval. Frames 808–1008 remain a final temporal holdout and are excluded from the reported development ranking. The 24 development endpoints are sampled deterministically between indices 731 and 807. The input window has 24 steps. Mean and population standard deviation use only training-frame Ki observations. The original outer-hidden and dense labels were excluded from model development.

The comparison uses seeds 505, 606 and 707, yielding 54 conditions per learned model. Paired controls and the four-cell ablation use seeds 1103, 1201 and 1301. These are separate result sets. Dataset-level scores average folds and seeds equally. The six-dataset macro averages the dataset scores equally. Deterministic kriging has three folds per dataset and no training-seed repetition.

### Model and training configuration

- Hidden width 64, three KITS graph convolutions, reference feature transfer and fusion.
- Connection coefficient 0.5 and distance-adjacency threshold 0.1. Same-room edge augmentation is disabled. The inherited wall-correction parameter is fixed at zero. Component relationships enter the kriging kernel and actual-support gate.
- Exponential component kernel with length equal to the public median positive distance times its selected multiplier. Donor baselines exclude the donor itself. The RMS scale has a lower bound of 0.1 and numerical stabilizer 1e-6.
- Two zero-initialized 64-to-1 heads, each with 65 parameters. The calibration head activates only when the query has zero actually visible donors in its public component.
- Replicated temporal boundary padding. Native virtual-node training, graph-local virtual baselines, cycle consistency and the 12-step residual forecasting decoder are retained.
- 48 epochs, 128 sampled examples per epoch, batch size 16 and 384 Adam updates. Learning rate 0.001. Parent and calibration gradients are clipped separately to norm 5. The wall scalar has learning rate zero.
- Parent objective weights are reconstruction 1, cycle 0.25, forecast 0.04, query-history 1 and query-endpoint 3. The auxiliary whole-component-masked objective is history L1 plus three times endpoint L1. It updates only the calibration head through detached features, reference predictions and scale.
- Development evaluation occurs every two epochs. The selected checkpoint minimizes development MAE within the 48-epoch budget. These results establish fixed-budget performance.

The per-condition archived sampling plans are supplied in `configs/plans`. They are preferred for exact reproduction. Fresh sampling follows the original seeded two-block schedule in `brick/training.py`.

```sh
python -m brick train --scene Hospital_2 --fold 0 --seed 505 --device cuda --output outputs/train_hospital2
```

The code exports the final model recipe. Opponent implementations remain separate studies. Opponent-specific fixed settings included GP-VAE learning rate 0.0003, KCN learning rate 0.005 with weight decay 0.0005, CSDI 50 diffusion steps, and PriSTI 100 diffusion steps with effective batch size 16 through microbatch accumulation. Diffusion point predictions use the median of three trajectories. DCRNN uses a one-step current-query decoder, NAOMI uses a single missing spatial token between observed donor anchors, and AGCRN uses query and distance-sorted donor slots. These are declared task adaptations.

### Fixed kernel settings

Each tuple contains length multiplier, component penalty and nugget.

| 数据集 | 空间折0 / (a, β, ν) | 空间折1 / (a, β, ν) | 空间折2 / (a, β, ν) |
|---|---|---|---|
| Hospital_2 | 4 / 3 / 1e-05 | 4 / 3 / 1e-05 | 4 / 3 / 1e-05 |
| Office_1 | 1 / 0.5 / 1e-05 | 1 / 0.5 / 1e-05 | 1 / 0.5 / 1e-05 |
| Residence_2 | 1 / 3 / 0.1 | 1 / 3 / 0.1 | 1 / 3 / 0.1 |
| Office_2 | 4 / 3 / 1e-05 | 4 / 6 / 1e-05 | 4 / 6 / 1e-05 |
| Residence_1 | 0.5 / 6 / 0.01 | 4 / 6 / 1e-05 | 4 / 6 / 1e-05 |
| Residence_3 | 4 / 3 / 1e-05 | 4 / 6 / 0.01 | 4 / 6 / 0.01 |

Hospital_2, Office_1 and Residence_2 selected one shared building setting from 60 candidates using three-fold development-query MAE. The grids were length multipliers {0.25, 0.5, 1, 2, 4}, component penalties {0, 0.5, 1.5, 3}, and nuggets {1e-5, 0.01, 0.1}. Office_2, Residence_1 and Residence_3 selected per-fold settings by training-Ki leave-one-out MAE from 18 candidates, with grids {0.5, 1, 4}, {0, 3, 6} and {1e-5, 0.01}. BRICK and geometry-sharing controls use the corresponding same setting.

## Six-dataset comparison

The following table reports per-dataset MAE and macro MAE/RMSE in degrees Celsius. All 23 implementations are retained, including the standalone component-conditioned kriging baseline. Per-condition MAE and RMSE are available in `results/comparison_conditions.csv`.

| Method | Hospital_2 | Office_1 | Residence_2 | Office_2 | Residence_1 | Residence_3 | Macro MAE | Macro RMSE |
|---|---|---|---|---|---|---|---|---|
| IGNNK | 3.971712 | 0.399199 | 1.663945 | 1.048636 | 2.172440 | 3.365814 | 2.103624 | 2.789847 |
| SAITS | 4.343037 | 0.590697 | 1.479869 | 1.043604 | 1.811462 | 3.734159 | 2.167138 | 2.862866 |
| BRITS | 4.820626 | 0.618923 | 1.413518 | 1.213142 | 1.819623 | 4.001321 | 2.314526 | 3.062375 |
| SATCN | 4.415993 | 0.583708 | 1.152579 | 0.909629 | 2.711923 | 3.459307 | 2.205523 | 2.876103 |
| STGCN | 4.059552 | 0.459606 | 1.402151 | 0.994001 | 1.897338 | 3.114031 | 1.987780 | 2.652661 |
| Graph-WaveNet | 3.845000 | 0.488170 | 1.393372 | 0.920881 | 2.249931 | 3.408341 | 2.050949 | 2.758914 |
| MTGNN | 4.558141 | 0.605615 | 1.479855 | 0.945704 | 1.926209 | 3.561946 | 2.179578 | 2.878800 |
| DualSTN | 5.220513 | 0.477490 | 1.944593 | 1.193664 | 2.012479 | 3.581158 | 2.404983 | 3.176590 |
| Senseiver | 4.682777 | 0.828192 | 1.702991 | 1.224568 | 2.008761 | 3.943104 | 2.398399 | 3.104551 |
| GRIN | 4.605406 | 0.510948 | 1.412326 | 0.939719 | 2.357244 | 3.381558 | 2.201200 | 2.883833 |
| ImputeFormer | 4.445713 | 0.640181 | 1.505474 | 0.941550 | 2.070632 | 3.411577 | 2.169188 | 2.884490 |
| KCN | 5.411218 | 0.640548 | 1.458546 | 1.090460 | 2.175803 | 3.311507 | 2.348014 | 2.914974 |
| CSDI | 6.194473 | 1.081605 | 3.134784 | 1.879344 | 2.659586 | 5.776497 | 3.454382 | 4.826913 |
| AGCRN | 4.036161 | 0.560179 | 1.785128 | 0.975232 | 2.064384 | 3.290587 | 2.118612 | 2.779241 |
| BL-KITS | 4.720889 | 0.596474 | 1.575365 | 1.051608 | 2.169274 | 3.520811 | 2.272404 | 3.004812 |
| DCRNN | 4.259529 | 0.481797 | 1.465737 | 0.920412 | 1.827128 | 3.293023 | 2.041271 | 2.730094 |
| GP-VAE | 5.621406 | 0.624684 | 1.723072 | 1.458459 | 2.230207 | 3.858535 | 2.586061 | 3.131887 |
| NAOMI | 3.955914 | 0.517159 | 1.487208 | 0.944570 | 1.977033 | 3.759458 | 2.106890 | 2.885228 |
| INCREASE | 4.459227 | 0.693430 | 1.473104 | 0.979623 | 2.119240 | 3.373166 | 2.182965 | 2.904105 |
| PriSTI | 5.706918 | 0.818942 | 2.648906 | 1.690331 | 2.233195 | 4.527900 | 2.937699 | 3.877534 |
| BRICK | 0.723918 | 0.391672 | 1.150054 | 0.355108 | 1.686186 | 2.756147 | 1.177181 | 1.654835 |
| 普通克里金 | 4.865944 | 0.426543 | 1.622911 | 0.903904 | 3.197838 | 3.263825 | 2.380161 | 3.146945 |
| 组件克里金 | 0.980200 | 0.422594 | 1.402169 | 0.596353 | 2.068462 | 3.062387 | 1.422028 | 1.946591 |

BRICK ranks first by the six-dataset macro MAE and RMSE. Its macro MAE is 1.177180948 and macro RMSE is 1.654835397. Its macro MAE improvement over component-conditioned kriging is 17.218%. Individual building and condition exceptions are retained in the condition table.

## Paired controls and detailed ablation

The following controls share the geometry residual view. The same-view GCN also receives the auxiliary masked examples, while the isolated-head GCN restricts auxiliary supervision to an independent head.

| 数据集 | BRICK / MAE / RMSE | 普通GCN / MAE / RMSE | 同视图GCN / MAE / RMSE | 隔离头GCN / MAE / RMSE |
|---|---|---|---|---|
| Hospital_2 | 0.7401 / 1.2166 | 0.8415 / 1.2949 | 0.7334 / 1.2338 | 0.7334 / 1.2309 |
| Office_1 | 0.3992 / 0.5245 | 0.3922 / 0.5051 | 0.3747 / 0.4866 | 0.4038 / 0.5123 |
| Residence_2 | 1.1511 / 1.6690 | 1.2997 / 1.8738 | 1.2875 / 1.8517 | 1.2984 / 1.7994 |
| Office_2 | 0.3438 / 0.5222 | 0.5148 / 0.8584 | 0.5162 / 0.8435 | 0.3600 / 0.6271 |
| Residence_1 | 1.6480 / 2.2045 | 1.9095 / 2.4566 | 1.8557 / 2.3842 | 1.9129 / 2.4280 |
| Residence_3 | 2.7177 / 3.6579 | 2.6732 / 3.5519 | 2.9038 / 3.8295 | 2.5459 / 3.3870 |
| 六数据集 / 均值±标准差 | 1.1667±0.9040 / 1.6324±1.1886 | 1.2718±0.8835 / 1.7568±1.1237 | 1.2786±0.9664 / 1.7716±1.2181 | 1.2091±0.8816 / 1.6641±1.1099 |

The four-cell study defines A as the complete residual representation and B as the complete dual-path inference module. All four cells share backbone, partitions, seeds, sample plans and optimization budget.

| 数据集 | 基线 / MAE / RMSE | 基线+A / MAE / RMSE | 基线+B / MAE / RMSE | 完整BRICK / MAE / RMSE |
|---|---|---|---|---|
| Hospital_2 | 4.3622 / 6.3698 | 0.9645 / 1.5833 | 4.2411 / 6.7760 | 0.7401 / 1.2166 |
| Office_1 | 0.6306 / 0.8830 | 0.3992 / 0.5045 | 0.6250 / 0.9012 | 0.3992 / 0.5245 |
| Residence_2 | 1.5729 / 2.0199 | 1.2020 / 1.7307 | 1.5289 / 2.0629 | 1.1511 / 1.6690 |
| Office_2 | 0.9380 / 1.4848 | 0.5615 / 0.8516 | 0.9616 / 1.4438 | 0.3438 / 0.5222 |
| Residence_1 | 2.1445 / 2.6357 | 1.9533 / 2.5162 | 2.1944 / 2.6406 | 1.6480 / 2.2045 |
| Residence_3 | 3.3091 / 3.9485 | 2.9614 / 3.9534 | 3.3598 / 3.9935 | 2.7177 / 3.6579 |

The baseline, A-only, B-only and A+B macro MAEs are 2.159544888, 1.340304898, 2.151789157 and 1.166653222. Adding B to A reduces MAE by 12.956% and RMSE by 12.075%. B alone increases macro RMSE from 2.890270878 to 2.969648813. Office_1 slightly favours A alone. The evidence supports an aggregate benefit of the combined modules and preserves these exceptions.

### Actual observation support

| 范围 | 支持状态 | 局部读出父配方 | BRICK | 隔离头GCN |
|---|---|---|---|---|
| H2/O1/R2 | 同组件无支持 | 1.306896 | 0.904364 | 0.727318 |
| H2/O1/R2 | 同组件有支持 | 0.745062 | 0.772402 | 0.834619 |
| O2/R1/R3 | 同组件无支持 | 3.405754 | 2.855643 | 2.720479 |
| O2/R1/R3 | 同组件有支持 | 0.492681 | 0.437034 | 0.617499 |

The parent local-readout recipe is an ablation variant. Complete BRICK improves its no-support strata in both building groups. The isolated-head GCN has smaller errors in these strata. Empty strata are excluded within each aggregation and are recorded in the original analysis.

## Frozen cross-building evaluation

Hospital_1 and Residence_4 were evaluated with frozen source models, source normalization and source kernel settings. For each of seeds 1103, 1201 and 1301, nine source experts from Hospital_2, Office_1 and Residence_2 and their three folds were averaged equally after conversion to Celsius. This aggregation was chosen before target values were accessed. Target training, optimization and hyperparameter selection were all zero. Predictions were sealed before a separate query-endpoint scorer was invoked.

| 方法 | Hospital_1 | Residence_4 | 两建筑平均MAE |
|---|---|---|---|
| BRICK | 1.591379 | 2.172898 | 1.882138 |
| 普通GCN | 1.643260 | 2.293533 | 1.968396 |
| 同视图GCN | 1.571068 | 2.525345 | 2.048207 |
| 隔离头GCN | 1.637809 | 2.472358 | 2.055084 |
| 源冻结组件克里金 | 1.369068 | 2.557098 | 1.963083 |

BRICK has the lowest two-building macro MAE. Hospital_1 favours component-conditioned kriging and the same-view GCN. The evaluation used two target buildings and the fixed nine-expert aggregation. Hospital_1 had historical hide20 training exposure on 2026-09-24. A complete history establishing Residence_4 as previously unused has not been certified. The frozen evaluation is reported with that provenance.

## Attribution and license

The graph backbone derives from KITS, AAAI 2025. Its license is retained in `LICENSE`. The S-Lab License 1.0 permits use and redistribution under its stated non-commercial conditions and requires the specified notices. Commercial use requires the applicable contributor permission. Data provenance and version identifiers are recorded separately in `data/datasets.json`.

## Whole-floor reconstruction figures

The igures directory contains the two manuscript cases and complete 23-method atlases for the retained Office_2 and residential cases. All panels use the same global inverse-distance interpolation and colour range within each case. Circles mark observations and triangles mark queries. Cases were selected retrospectively from archived development predictions.
