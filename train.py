import warnings
import copy
import datetime
import os
import sys
import pathlib
import pandas as pd
from argparse import ArgumentParser

import random
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from lib import fillers, datasets, config
from lib.data.datamodule import SpatioTemporalDataModule
from lib.data.imputation_dataset import (ImputationDataset, GraphImputationDataset,
                                         ForecastImputationDataset, GraphForecastImputationDataset)
from lib.nn import models
from lib.nn.utils.metric_base import MaskedMetric
from lib.nn.utils.metrics import MaskedMAE, MaskedMAPE, MaskedMSE, MaskedMRE
from lib.utils import parser_utils, numpy_metrics, ensure_list, prediction_dataframe
from lib.utils.parser_utils import str_to_bool


def has_graph_support(model_cls):
    return model_cls in [models.KITS]


def get_model_classes(model_str):
    if model_str == 'kits':
        model, filler = models.KITS, fillers.GCNCycVirtualFiller
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name, miss_rate=0.5, mode="road", test_entries="", **kwargs):
    if dataset_name[:3] == 'aqi':
        dataset = datasets.AirQuality(impute_nans=True, small=dataset_name[3:] == '36', p=miss_rate)
    elif dataset_name == 'la_point':
        dataset = datasets.MissingValuesMetrLA(p_fault=0., p_noise=miss_rate, mode=mode, test_entries=test_entries)
    elif dataset_name == 'bay_point':
        dataset = datasets.MissingValuesPemsBay(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'pems07_point':
        dataset = datasets.MissingValuesPems07(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'sea_loop_point':
        dataset = datasets.MissingValuesSeaLoop(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'nrel_al_point':
        dataset = datasets.MissingValuesNrelAl(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'nrel_md_point':
        dataset = datasets.MissingValuesNrelMd(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'ushcn':
        dataset = datasets.MissingValuesUshcn(p_fault=0., p_noise=miss_rate, mode=mode)
    elif dataset_name == 'intel_lab':
        variable = kwargs.get('variable', 'temperature')
        mask_mode = kwargs.get('mask_mode', 'road')
        holdout_rooms = kwargs.get('holdout_rooms', None)
        room_temp_offset = kwargs.get('room_temp_offset', None)
        split_file = kwargs.get('split_file', None)
        semisynth_file = kwargs.get('semisynth_file', None)
        dataset = datasets.IntelLabDataset(impute_nans=True, p=miss_rate,
                                           variable=variable,
                                           mask_mode=mask_mode,
                                           holdout_rooms=holdout_rooms,
                                           room_temp_offset=room_temp_offset,
                                           split_file=split_file,
                                           semisynth_file=semisynth_file)
    else:
        raise ValueError(f"Dataset {dataset_name} not available in this setting.")
    return dataset


def parse_args():
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument("--model-name", type=str, default='kits')
    parser.add_argument("--dataset-name", type=str, default='la_point')
    parser.add_argument("--miss-rate", default=0.5, type=float)
    parser.add_argument("--mode", default="road", choices=["road"], type=str)
    parser.add_argument("--test-entries", default="", choices=["", "metr_la_coarse_to_fine.txt", "metr_la_coarse_to_fine_hard.txt", "metr_la_region.txt", "metr_la_region_hard.txt"], type=str)
    parser.add_argument("--config", type=str, default="config/kits/la_point.yaml")
    parser.add_argument("--use-adj-drop", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--use-init", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--pretrained-model", type=str, default="")
    # Splitting/aggregation params
    parser.add_argument('--in-sample', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)
    parser.add_argument('--aggregate-by', type=str, default='mean')
    # Training params
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--scaled-target', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--grad-clip-val', type=float, default=1.)
    parser.add_argument('--grad-clip-algorithm', type=str, default='norm')
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument('--whiten-prob', type=float, default=0.05)
    parser.add_argument('--checkpoint-save-top-k', type=int, default=1)
    # graph params
    parser.add_argument("--adj-threshold", type=float, default=0.1)
    # PI-KITS params (Phase 2 & 3)
    parser.add_argument("--use-multi-relation", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--wall-lambda", type=float, default=0.5)
    parser.add_argument("--room-boost-factor", type=float, default=1.0)
    parser.add_argument("--add-same-room-edges", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--use-physics-loss", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--physics-loss-weight", type=float, default=0.1)
    parser.add_argument("--spatial-smooth-weight", type=float, default=0.01)
    # PI-KITS params (Phase 4: Temporal GRU)
    parser.add_argument("--use-temporal-gru", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--gru-num-layers", type=int, default=1)
    # PI-KITS params (Phase 5: Wall-Aware Attention)
    parser.add_argument("--use-wall-attention", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--wall-attn-heads", type=int, default=4)
    parser.add_argument("--wall-attn-dropout", type=float, default=0.1)
    # PI-KITS: Wall-Aware Hard Transfer (zero-param, inject wall_decay into cosine sim)
    parser.add_argument("--use-wall-transfer", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--wall-transfer-lambda", type=float, default=0.5,
                        help="Wall decay lambda for Hard Transfer (separate from adj wall_lambda)")
    # PI-KITS params (Phase 6: Physics-Guided Conv)
    parser.add_argument("--use-physics-conv", type=str_to_bool, nargs='?', const=True, default=False)
    # PI-KITS params (Phase 7: Multi-Scale Temporal)
    parser.add_argument("--use-multiscale-temporal", type=str_to_bool, nargs='?', const=True, default=False)
    # PI-KITS params (Phase 8: Uncertainty)
    parser.add_argument("--use-uncertainty", type=str_to_bool, nargs='?', const=True, default=False)
    # PI-KITS params (Phase 9: Multi-variable)
    parser.add_argument("--variable", type=str, default='temperature',
                        choices=['temperature', 'humidity', 'both'])
    # PI-KITS params (Phase 11: Forecasting)
    parser.add_argument("--forecast-horizon", type=int, default=0,
                        help="Number of future timesteps to predict (0=disabled)")
    parser.add_argument("--forecast-loss-weight", type=float, default=0.5,
                        help="Weight for forecast loss term")
    parser.add_argument("--forecast-detach", type=str_to_bool, nargs='?', const=True, default=True,
                        help="Detach encoder hidden before forecast decoder (default True)")
    parser.add_argument("--forecast-use-orig-adj", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Use original distance adj for forecast decoder (default False)")
    parser.add_argument("--freeze-encoder", type=str_to_bool, nargs='?', const=True, default=False,
                        help="Freeze encoder, only train forecast decoder (two-stage Stage2)")
    parser.add_argument("--stage2-from", type=str, default="",
                        help="Load Stage1 checkpoint for two-stage training (load weights then continue training)")
    # Evaluation mode params
    parser.add_argument("--mask-mode", type=str, default='road',
                        choices=['road', 'room_holdout'],
                        help="Mask mode: 'road' (random columns) or 'room_holdout' (whole rooms)")
    parser.add_argument("--holdout-rooms", type=int, nargs='*', default=None,
                        help="Room IDs to holdout when mask_mode='room_holdout'")
    parser.add_argument("--room-temp-offset", type=str, default=None,
                        help="Room temperature offset, e.g. '1:15,2:15' for +15°C to Room 1&2")
    parser.add_argument("--split-file", type=str, default="",
                        help="NPZ file with known_mask/hidden_mask for a fixed sensor split")
    parser.add_argument("--semisynth-file", type=str, default="",
                        help="NPZ file with a precomputed semi-synthetic temperature array")

    known_args, _ = parser.parse_known_args()
    model_cls, _ = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)

    pl.seed_everything(args.seed)

    # Parse room_temp_offset string (e.g., "1:15,2:15") into dict
    room_temp_offset = None
    offset_str = getattr(args, 'room_temp_offset', None)
    if offset_str:
        room_temp_offset = {}
        for part in offset_str.split(','):
            rid, val = part.split(':')
            room_temp_offset[int(rid)] = float(val)

    model_cls, filler_cls = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name, args.miss_rate, args.mode, args.test_entries,
                          variable=getattr(args, 'variable', 'temperature'),
                          mask_mode=getattr(args, 'mask_mode', 'road'),
                          holdout_rooms=getattr(args, 'holdout_rooms', None),
                          room_temp_offset=room_temp_offset,
                          split_file=getattr(args, 'split_file', ''),
                          semisynth_file=getattr(args, 'semisynth_file', ''))

    ########################################
    # create logdir and save configuration #
    ########################################
    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(config['logs'], args.dataset_name, args.model_name, exp_name)
    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################
    # instantiate dataset
    forecast_horizon = getattr(args, 'forecast_horizon', 0)
    if forecast_horizon > 0:
        dataset_cls = GraphForecastImputationDataset if has_graph_support(model_cls) else ForecastImputationDataset
        torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                    forecast_horizon=forecast_horizon,
                                    mask=dataset.training_mask,
                                    eval_mask=dataset.eval_mask,
                                    window=args.window,
                                    stride=args.stride)
    else:
        dataset_cls = GraphImputationDataset if has_graph_support(model_cls) else ImputationDataset
        torch_dataset = dataset_cls(*dataset.numpy(return_idx=True),
                                    mask=dataset.training_mask,
                                    eval_mask=dataset.eval_mask,
                                    window=args.window,
                                    stride=args.stride)

    # get train/val/test indices
    split_conf = parser_utils.filter_function_args(args, dataset.splitter, return_dict=True)
    train_idxs, val_idxs, test_idxs = dataset.splitter(torch_dataset, **split_conf)

    # configure datamodule
    data_conf = parser_utils.filter_args(args, SpatioTemporalDataModule, return_dict=True)
    if args.dataset_name in ["pems07_point"]:
        data_conf["scaling_type"] = "minmax"
        dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                      **data_conf)
        min_val = 0
        max_val = 1500
        print("Min Max Scaler - max: {}".format(max_val))
        dm.setup(min=min_val, max=max_val)
    elif args.dataset_name in ["nrel_al_point", "nrel_md_point"]:
        print("Use capacities as Min Max Scaler")
        data_conf["scaling_type"] = "minmax"
        dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                      **data_conf)
        files_info = pd.read_pickle('datasets/{}/nrel_file_infos.pkl'.format(args.dataset_name.replace("_point", "")))
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
        capacities = np.expand_dims(capacities, axis=(0, -1))
        min_val = np.zeros_like(capacities)
        dm.setup(min=min_val, max=capacities)
    else:
        dm = SpatioTemporalDataModule(torch_dataset, train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=test_idxs,
                                      **data_conf)
        dm.setup()

    # get adjacency matrix
    if args.dataset_name == 'intel_lab' and hasattr(args, 'use_multi_relation'):
        adj = dataset.get_similarity(thr=args.adj_threshold,
                                     use_multi_relation=args.use_multi_relation,
                                     wall_lambda=args.wall_lambda,
                                     room_boost_factor=getattr(args, 'room_boost_factor', 1.0),
                                     add_same_room_edges=getattr(args, 'add_same_room_edges', False),
                                     conn_lambda=getattr(args, 'conn_lambda', 0.0))
    else:
        adj = dataset.get_similarity(thr=args.adj_threshold)
    # force adj with no self loop
    np.fill_diagonal(adj, 0.)

    # Pass raw adjacency components for learnable conn_lambda (D1)
    if getattr(args, 'learnable_conn_lambda', False) and hasattr(dataset, '_adj_components'):
        comps = dataset._adj_components
        args._adj_dist_raw = comps['A_dist']
        args._adj_conn_norm = comps['A_conn']
        args._adj_room_raw = comps['A_room']
        print(f"[PI-KITS] Learnable conn: raw matrices passed to model")
    else:
        args._adj_dist_raw = None
        args._adj_conn_norm = None
        args._adj_room_raw = None

    # D3: Room embedding — pass room_ids to model
    if getattr(args, 'use_room_embed', False) and args.dataset_name == 'intel_lab':
        from lib.datasets.intel_lab import IntelLabDataset
        room_groups = IntelLabDataset.ROOM_GROUPS
        n_sensors = adj.shape[0]
        room_ids = np.zeros(n_sensors, dtype=np.int64)
        for room_id, sensor_list in room_groups.items():
            for s in sensor_list:
                if s < n_sensors:
                    room_ids[s] = room_id
        args._room_ids = room_ids
        print(f"[PI-KITS] Room IDs: {np.bincount(room_ids)} sensors per room")
    else:
        args._room_ids = None

    # Load walls matrix for Wall-Aware Attention or Wall-Aware Hard Transfer
    need_walls = (getattr(args, 'use_wall_attention', False) or
                  getattr(args, 'use_wall_transfer', False))
    if args.dataset_name == 'intel_lab' and need_walls:
        walls_path = os.path.join('datasets', 'intel_lab', 'walls.npy')
        if os.path.exists(walls_path):
            walls_matrix = np.load(walls_path)
            args._walls_matrix = walls_matrix
            print(f"[PI-KITS] Walls matrix loaded: {walls_matrix.shape}")
        else:
            args._walls_matrix = None
            print("[PI-KITS] Warning: walls.npy not found, wall features disabled")
    else:
        args._walls_matrix = None

    # Load conn matrix for Conn-Guided Donor Selection or Conn Loss
    need_conn = (getattr(args, 'use_conn_donor', False) or
                 getattr(args, 'use_conn_loss', False))
    if args.dataset_name == 'intel_lab' and need_conn:
        conn_path = os.path.join('datasets', 'intel_lab', 'conn.npy')
        if os.path.exists(conn_path):
            conn_matrix = np.load(conn_path)
            args._conn_matrix = conn_matrix
            print(f"[PI-KITS] Conn matrix loaded: {conn_matrix.shape}")
        else:
            args._conn_matrix = None
            print("[PI-KITS] Warning: conn.npy not found")
        # Also load dist for ConnSpatialLoss normalization
        if getattr(args, 'use_conn_loss', False):
            dist_path = os.path.join('datasets', 'intel_lab', 'dist.npy')
            if os.path.exists(dist_path):
                args._dist_matrix = np.load(dist_path)
                print(f"[PI-KITS] Dist matrix loaded for conn loss")
            else:
                args._dist_matrix = None
        else:
            args._dist_matrix = None
    else:
        args._conn_matrix = None
        args._dist_matrix = None

    # Compute original distance-only adjacency for forecast decoder (Approach B)
    if getattr(args, 'forecast_use_orig_adj', False) and args.dataset_name == 'intel_lab':
        adj_orig = dataset.get_similarity(thr=args.adj_threshold,
                                          use_multi_relation=False)
        np.fill_diagonal(adj_orig, 0.)
        args._adj_original = adj_orig
        print(f"[PI-KITS] Original distance adjacency computed for forecast decoder: "
              f"edges={int((adj_orig > 0).sum())}")
    else:
        args._adj_original = None

    # data = dataset.numpy()
    # train_mask = dataset.training_mask
    # test_mask = dataset.eval_mask
    #
    # train_y = data[dm.train_slice, ...]
    # train_m = train_mask[dm.train_slice, ...]
    # val_y = data[dm.val_slice, ...]
    # val_m = test_mask[dm.val_slice, ...]
    # test_y = data[dm.test_slice, ...]
    # test_m = test_mask[dm.test_slice, ...]
    #
    # if not os.path.exists("data"):
    #     os.mkdir("data")
    # data_name = ""
    # if "ushcn" in args.dataset_name:
    #     data_name = "USHCN"
    # base_path = "data/{}/".format(data_name)
    # if not os.path.exists(base_path):
    #     os.mkdir(base_path)
    # base_path = "{}{}/".format(base_path, args.mode)
    # if not os.path.exists(base_path):
    #     os.mkdir(base_path)
    # if not os.path.exists("{}train_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else "")):
    #     np.save("{}train_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), train_y)
    #     np.save("{}train_m_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), train_m)
    #     np.save("{}val_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), val_y)
    #     np.save("{}val_m_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), val_m)
    #     np.save("{}test_y_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), test_y)
    #     np.save("{}test_m_{}_seed{}{}.npy".format(base_path, args.miss_rate, args.seed, "_{}".format(args.test_entries.split(".")[0]) if args.test_entries != "" else ""), test_m)
    #
    # if not os.path.exists("{}adj.npy".format(base_path, args.miss_rate)):
    #     np.save("{}adj.npy".format(base_path, args.miss_rate), adj)
    # sys.exit(0)

    # # ========================================
    # # for scalability testing purpose
    # print("Original adj shape:", adj.shape)
    # adj = np.repeat(adj, 100, axis=0)
    # adj = np.repeat(adj, 100, axis=1)
    # print("Scaled adj shape:", adj.shape)
    # # ========================================

    ########################################
    # predictor                            #
    ########################################
    # model's inputs
    additional_model_hparams = dict(adj=adj, d_in=dm.d_in, n_nodes=dm.n_nodes, args=args)
    model_kwargs = parser_utils.filter_args(args={**vars(args), **additional_model_hparams},
                                            target_cls=model_cls,
                                            return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(F, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {
        'mae': MaskedMAE(compute_on_step=False),
        'mape': MaskedMAPE(compute_on_step=False),
        'mse': MaskedMSE(compute_on_step=False),
        'mre': MaskedMRE(compute_on_step=False)
    }

    # filler's inputs
    scheduler_class = CosineAnnealingLR if args.use_lr_schedule else None
    additional_filler_hparams = dict(model_class=model_cls,
                                     model_kwargs=model_kwargs,
                                     optim_class=torch.optim.Adam,
                                     optim_kwargs={'lr': args.lr,
                                                   'weight_decay': args.l2_reg},
                                     loss_fn=loss_fn,
                                     metrics=metrics,
                                     scheduler_class=scheduler_class,
                                     scheduler_kwargs={
                                         'eta_min': 0.0001,
                                         'T_max': args.epochs
                                     }
                                     )
    filler_kwargs = parser_utils.filter_args(args={**vars(args), **additional_filler_hparams},
                                             target_cls=filler_cls,
                                             return_dict=True)
    filler = filler_cls(**filler_kwargs)

    # Two-stage training: load Stage1 checkpoint for Stage2
    stage2_from = getattr(args, 'stage2_from', '')
    if stage2_from and os.path.exists(stage2_from):
        state_dict = torch.load(stage2_from, lambda storage, loc: storage)['state_dict']
        missing, unexpected = filler.load_state_dict(state_dict, strict=False)
        print(f"[PI-KITS] Stage2: loaded {len(state_dict) - len(missing)} params from {stage2_from}")
        if missing:
            print(f"  New params (not in Stage1): {missing}")

    if args.pretrained_model == "" or args.pretrained_model == None:
        ########################################
        # training                             #
        ########################################
        # callbacks
        early_stop_callback = EarlyStopping(monitor='val_mae', patience=args.patience, mode='min', verbose=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=logdir,
            save_top_k=args.checkpoint_save_top_k,
            monitor='val_mae',
            mode='min')

        logger = TensorBoardLogger(logdir, name="model")

        trainer = pl.Trainer(max_epochs=args.epochs,
                             logger=logger,
                             default_root_dir=logdir,
                             accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                             devices=1,
                             gradient_clip_val=args.grad_clip_val,
                             gradient_clip_algorithm=args.grad_clip_algorithm,
                             callbacks=[early_stop_callback, checkpoint_callback])

        trainer.fit(filler, datamodule=dm)

        trainer.test(datamodule=dm)

        filler.load_state_dict(torch.load(checkpoint_callback.best_model_path,
                                          lambda storage, loc: storage)['state_dict'])
    else:
        state_dict = torch.load(args.pretrained_model, lambda storage, loc: storage)['state_dict']
        adj = torch.from_numpy(adj)
        state_dict["model.adj"] = adj  # in case of using pretrained model of other datasets to infer current dataset
        filler.load_state_dict(state_dict)

    ########################################
    # testing                              #
    ########################################
    filler.freeze()
    filler.eval()

    if torch.cuda.is_available():
        filler.cuda()

    with torch.no_grad():
        y_true, y_hat, mask = filler.predict_loader(dm.test_dataloader(), return_mask=True)

    is_multivar = (y_hat.shape[-1] > 1)
    eval_mask = dataset.eval_mask[dm.test_slice]
    df_true = dataset.df.iloc[dm.test_slice]

    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mape': numpy_metrics.masked_mape,
        'mre': numpy_metrics.masked_mre,
        'mse': numpy_metrics.masked_mse,
        'r2': numpy_metrics.masked_r2
    }

    if is_multivar:
        # Multi-variable: evaluate each channel separately
        var_names = ['temperature', 'humidity']
        y_hat_np = y_hat.detach().cpu().numpy()  # (samples, window, N, 2)
        for ch, var_name in enumerate(var_names):
            y_hat_ch = y_hat_np[..., ch]  # (samples, window, N)
            eval_mask_ch = eval_mask[..., ch] if eval_mask.ndim == 3 else eval_mask
            print(f'\n=== {var_name.upper()} ===')
            # Aggregate predictions
            index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
            aggr_methods = ensure_list(args.aggregate_by)
            df_hats = prediction_dataframe(y_hat_ch, index, dataset.df.columns, aggregate_by=aggr_methods)
            df_hats = dict(zip(aggr_methods, df_hats))
            for aggr_by, df_hat in df_hats.items():
                print(f'- AGGREGATE BY {aggr_by.upper()}')
                for metric_name, metric_fn in metrics.items():
                    error = metric_fn(df_hat.values, df_true.values, eval_mask_ch).item()
                    print(f' {metric_name}: {error:.4f}')
                    if metric_name == "mse":
                        print(f'rmse: {np.sqrt(error):.4f}')
    else:
        # Single-variable: original logic
        y_hat = y_hat.detach().squeeze(-1).cpu().numpy()
        # Aggregate predictions in dataframes
        index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
        aggr_methods = ensure_list(args.aggregate_by)
        df_hats = prediction_dataframe(y_hat, index, dataset.df.columns, aggregate_by=aggr_methods)
        df_hats = dict(zip(aggr_methods, df_hats))
        for aggr_by, df_hat in df_hats.items():
            # Compute error
            print(f'- AGGREGATE BY {aggr_by.upper()}')
            for metric_name, metric_fn in metrics.items():
                error = metric_fn(df_hat.values, df_true.values, eval_mask).item()
                print(f' {metric_name}: {error:.4f}')
                if metric_name == "mse":
                    print(f'rmse: {np.sqrt(error):.4f}')

    # Save predictions for visualization
    save_predictions(y_true, y_hat, mask, logdir, dataset, dm, filler)

    # Phase 11: Forecast evaluation (using best checkpoint)
    forecast_horizon = getattr(args, 'forecast_horizon', 0)
    if forecast_horizon > 0:
        print(f'\n=== FORECAST EVALUATION (horizon={forecast_horizon}) ===')
        try:
            from lib.fillers.filler import move_data_to_device
            from lib import epsilon as lib_eps
            fc_preds = []
            fc_trues = []
            with torch.no_grad():
                for batch in dm.test_dataloader():
                    batch = move_data_to_device(batch, filler.device)
                    batch_data, batch_preprocessing = filler._unpack_batch(batch)

                    # Extract targets BEFORE forward (pop from batch_data)
                    y_forecast = batch_data.pop('y_forecast', None)
                    _ = batch_data.pop('eval_mask', None)
                    _ = batch_data.pop('y')
                    _ = batch_data.pop('forecast_mask', None)

                    # Direct forward (avoid predict_batch double-unpack issues)
                    result = filler.forward(**batch_data)

                    # Unpack forecast from model eval output
                    forecast = None
                    if isinstance(result, (tuple, list)):
                        idx = 1  # skip imputation at result[0]
                        if filler.model.use_uncertainty and len(result) > idx:
                            idx += 1  # skip logvar
                        if filler.model.forecast_horizon > 0 and len(result) > idx:
                            forecast = result[idx]

                    # Inverse transform forecast (model output in scaled space)
                    # y_forecast is already in ORIGINAL space (same as y, never scaled by dataset)
                    scale = batch_preprocessing.get('scale', 1.)
                    bias = batch_preprocessing.get('bias', 0.)
                    trend = batch_preprocessing.get('trend', 0.)

                    # Debug: print shapes and values for first batch
                    if len(fc_preds) == 0:
                        print(f'  [DEBUG] scale={scale.shape if hasattr(scale,"shape") else scale}, '
                              f'bias mean={bias.mean().item() if hasattr(bias,"mean") else bias:.4f}, '
                              f'scale mean={scale.mean().item() if hasattr(scale,"mean") else scale:.4f}')
                        if forecast is not None:
                            print(f'  [DEBUG] forecast (scaled): '
                                  f'mean={forecast.mean():.4f}, range=[{forecast.min():.2f}, {forecast.max():.2f}]')
                        if y_forecast is not None:
                            print(f'  [DEBUG] y_forecast (original): '
                                  f'mean={y_forecast.mean():.4f}, range=[{y_forecast.min():.2f}, {y_forecast.max():.2f}]')

                    if forecast is not None:
                        forecast = forecast * (scale + lib_eps) + bias + trend
                        if len(fc_preds) == 0:
                            print(f'  [DEBUG] forecast AFTER postprocess: '
                                  f'mean={forecast.mean():.4f}')
                        fc_preds.append(forecast.detach().cpu())
                    if y_forecast is not None:
                        # y_forecast is already in original space — no postprocess needed
                        fc_trues.append(y_forecast.detach().cpu())

            if fc_preds and fc_trues:
                fc_pred = torch.cat(fc_preds, dim=0).numpy()
                fc_true = torch.cat(fc_trues, dim=0).numpy()
                print(f'  Forecast shape: pred={fc_pred.shape}, true={fc_true.shape}')

                for h in range(forecast_horizon):
                    pred_h = fc_pred[:, h, :, :]
                    true_h = fc_true[:, h, :, :]
                    mae_h = np.nanmean(np.abs(pred_h - true_h))
                    rmse_h = np.sqrt(np.nanmean((pred_h - true_h) ** 2))
                    print(f'  h={h+1:2d} | MAE={mae_h:.4f}  RMSE={rmse_h:.4f}')

                mae_all = np.nanmean(np.abs(fc_pred - fc_true))
                rmse_all = np.sqrt(np.nanmean((fc_pred - fc_true) ** 2))
                print(f'  ALL  | MAE={mae_all:.4f}  RMSE={rmse_all:.4f}')

                fc_path = os.path.join(logdir, 'forecast_predictions.npz')
                np.savez(fc_path, forecast_pred=fc_pred, forecast_true=fc_true)
                print(f'  Saved: {fc_path}')
            else:
                print(f'  [WARNING] No forecast outputs collected from model.')
        except Exception as e:
            import traceback
            print(f'  [WARNING] Forecast evaluation failed: {e}')
            traceback.print_exc()

    return y_true, y_hat, mask


def save_predictions(y_true, y_hat, mask, logdir, dataset, dm, filler):
    """保存预测结果供可视化使用"""
    save_dict = {
        'y_true': y_true.detach().cpu().numpy() if hasattr(y_true, 'detach') else y_true,
        'y_pred': y_hat.detach().cpu().numpy() if hasattr(y_hat, 'detach') else y_hat,
        'eval_mask': dataset.eval_mask[dm.test_slice] if dataset.eval_mask is not None else None,
    }

    # 尝试获取 logvar (不确定性)
    try:
        filler.freeze()
        filler.eval()
        if hasattr(filler, 'model') and hasattr(filler.model, 'use_uncertainty') and filler.model.use_uncertainty:
            from lib.fillers.filler import move_data_to_device
            logvars = []
            with torch.no_grad():
                for batch in dm.test_dataloader():
                    batch = move_data_to_device(batch, filler.device)
                    batch_data, batch_preprocessing = filler._unpack_batch(batch)
                    _ = batch_data.pop('eval_mask', None)
                    _ = batch_data.pop('y')
                    _ = batch_data.pop('y_forecast', None)
                    _ = batch_data.pop('forecast_mask', None)
                    y_hat_batch = filler.predict_batch(batch, preprocess=False, postprocess=True)
                    if isinstance(y_hat_batch, (list, tuple)) and len(y_hat_batch) >= 2:
                        logvars.append(y_hat_batch[1].detach().cpu())
            if logvars:
                save_dict['logvar'] = torch.cat(logvars, 0).numpy()
    except Exception as e:
        print(f"  [Warning] Could not extract logvar: {e}")

    # 保存时间戳
    try:
        index = dm.torch_dataset.data_timestamps(dm.testset.indices, flatten=False)['horizon']
        save_dict['timestamps'] = index
    except Exception:
        pass

    pred_path = os.path.join(logdir, 'predictions.npz')
    np.savez(pred_path, **{k: v for k, v in save_dict.items() if v is not None})
    print(f"\n[PI-KITS] Predictions saved: {pred_path}")
    return pred_path


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    run_experiment(args)
