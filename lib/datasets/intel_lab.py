"""
Intel Lab Dataset for KITS
基于 Intel Berkeley Research Lab 传感器网络数据
"""

import os
import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils.utils import thresholded_gaussian_kernel
from ..utils import sample_mask
from ..nn.layers.multi_relation_adj import build_multi_relation_adj


class IntelLabDataset(PandasDataset):
    """
    Intel Berkeley Research Lab 室内环境传感器数据集

    42个传感器节点 (剔除12个严重缺失节点)
    时序特征: 温度, 湿度, 或两者联合
    空间特征: 坐标、连接概率、墙体、房间
    """

    # 房间分组 (基于 room.npy connected components)
    # Room 0: 28 sensors (main lab)
    # Room 1: 1 sensor  (isolated, sensor #8)
    # Room 2: 13 sensors (secondary area)
    ROOM_GROUPS = {
        0: [0, 1, 2, 3, 4, 5, 6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41],
        1: [8],
        2: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    }

    def __init__(self, impute_nans=True, p=0.5, variable='temperature',
                 mask_mode='road', holdout_rooms=None, room_temp_offset=None,
                 split_file=None, semisynth_file=None):
        """
        Args:
            impute_nans: 是否用均值填充 NaN
            p: 模拟缺失传感器的比例 (用于评估)
            variable: 'temperature', 'humidity', 或 'both' (联合预测)
            mask_mode: 'road' (随机列遮罩) 或 'room_holdout' (整房间遮罩)
            holdout_rooms: mask_mode='room_holdout' 时，要遮罩的房间ID列表, 如 [2]
            room_temp_offset: dict, 如 {1: 15, 2: 15} 表示给Room1/2加15°C偏移
        """
        self.variable = variable
        self.mask_mode = mask_mode
        self.holdout_rooms = holdout_rooms or []
        self.room_temp_offset = room_temp_offset or {}
        self.split_file = split_file or None
        self.semisynth_file = semisynth_file or None
        self.eval_mask = None
        self.infer_eval_from = 'next'
        self.test_months = [3]

        df, dist, mask = self.load(impute_nans=impute_nans, p=p)
        self.dist = dist

        # 加载额外空间数据
        base_path = datasets_path['intel_lab']
        self.conn = np.load(os.path.join(base_path, 'conn.npy'))
        self.room = np.load(os.path.join(base_path, 'room.npy'))
        self.walls = np.load(os.path.join(base_path, 'walls.npy'))
        self.coords = np.load(os.path.join(base_path, 'coords.npy'))

        # 多变量模式: 存储3D数据以便numpy()返回
        self._multivar_data = None
        self._multivar_mask = None
        self._multivar_eval_mask = None

        if variable == 'both':
            # 存储 (T, N, 2) 数组供 numpy() 使用
            self._multivar_data = self._both_data
            self._multivar_mask = self._both_mask
            self._multivar_eval_mask = self._both_eval_mask
            # 清理临时属性
            del self._both_data, self._both_mask, self._both_eval_mask

        super().__init__(dataframe=df, u=None, mask=mask,
                         name='intel_lab', freq='30T', aggr='nearest')

        # 应用房间温度偏移 (半合成实验: 模拟跨房间大温差)
        # 渐进式偏移: 模拟HVAC启动过程
        #   Phase 1 (前20%时间): 原始温度
        #   Phase 2 (24小时过渡): 线性渐变, 模拟空调逐渐起效
        #   Phase 3 (剩余时间): 维持目标偏移
        if self.room_temp_offset and not self.semisynth_file and variable in ('temperature', 'both'):
            T = len(self.df)
            start_t = int(T * 0.2)        # 20%处开始
            ramp_steps = 48               # 48步=24小时(30min间隔)渐变

            # 构建时间轴上的偏移曲线: 0→target, 渐进过渡
            offset_curve = np.zeros(T)
            for t in range(start_t, T):
                progress = min(1.0, (t - start_t) / ramp_steps)
                offset_curve[t] = progress  # 0→1的渐变因子

            for room_id, offset in self.room_temp_offset.items():
                sensor_idxs = self.ROOM_GROUPS.get(int(room_id), [])
                if sensor_idxs:
                    cols = [self.df.columns[i] for i in sensor_idxs
                            if i < len(self.df.columns)]
                    # 每个时步加不同的偏移量 (渐进式)
                    self.df[cols] = self.df[cols].add(
                        offset * offset_curve, axis=0)
                    print(f"[Temp-Offset] Room {room_id}: +{offset}°C gradual "
                          f"(start={start_t}, ramp={ramp_steps} steps), "
                          f"{len(cols)} sensors")

        # 多变量模式: 用3D eval_mask覆盖2D版本
        if variable == 'both':
            self.eval_mask = self._multivar_eval_mask

    def load(self, impute_nans=True, p=0.5):
        """加载预处理后的数据"""
        base_path = datasets_path['intel_lab']
        h5_path = os.path.join(base_path, 'intel_lab.h5')

        if self.variable == 'both':
            return self._load_both(base_path, h5_path, impute_nans, p)

        # 单变量模式 (原始逻辑)
        df = pd.DataFrame(pd.read_hdf(h5_path, self.variable))

        semisynth = None
        if self.semisynth_file:
            if self.variable != 'temperature':
                raise ValueError("semisynth_file currently supports temperature only")
            semisynth = np.load(self.semisynth_file, allow_pickle=True)
            temperature = semisynth["temperature"]
            if temperature.shape != df.shape:
                raise ValueError(
                    f"semisynth temperature shape {temperature.shape} != source shape {df.shape}")
            df = pd.DataFrame(temperature, index=df.index, columns=df.columns)
            print(f"[SemiSynth] Loaded temperature from {self.semisynth_file}")

        if self.variable == 'temperature':
            mask_raw = np.load(os.path.join(base_path, 'mask_temp.npy'))
        else:
            mask_raw = np.load(os.path.join(base_path, 'mask_humid.npy'))

        if semisynth is not None and "natural_mask" in semisynth:
            mask = semisynth["natural_mask"].astype('uint8')
        else:
            mask = (~df.isna()).astype('uint8').values

        if self.split_file:
            known_mask = self._load_split_known_mask(mask.shape[1])
            eval_mask = np.tile((~known_mask).astype('uint8')[None, :],
                                (mask.shape[0], 1))
            print(f"[RoomCov Split] {self.split_file}: "
                  f"hidden={(~known_mask).sum()}/{mask.shape[1]}, "
                  f"known={known_mask.sum()}/{mask.shape[1]}")
        elif self.mask_mode == 'room_holdout' and self.holdout_rooms:
            # Room-holdout: 遮罩指定房间的所有传感器 (所有时间步)
            eval_mask = np.zeros(mask.shape, dtype='uint8')
            for room_id in self.holdout_rooms:
                sensor_idxs = self.ROOM_GROUPS.get(room_id, [])
                eval_mask[:, sensor_idxs] = 1
            n_holdout = eval_mask[0].sum()
            print(f"[Room-Holdout] Holding out rooms {self.holdout_rooms}: "
                  f"{n_holdout}/{mask.shape[1]} sensors masked")
        else:
            # 默认: 随机列遮罩
            eval_mask = sample_mask(mask.shape, p=0., p_noise=p, mode='road')
        self.eval_mask = (eval_mask & mask).astype('uint8')

        if impute_nans:
            hourly_mean = df.groupby(df.index.hour).transform('mean')
            df = df.fillna(hourly_mean)
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.fillna(df.mean())
            df = df.fillna(0)

        dist = np.load(os.path.join(base_path, 'dist.npy'))
        return df, dist, mask

    def _load_both(self, base_path, h5_path, impute_nans, p):
        """加载温度+湿度联合数据, 返回 (T, N, 2) 格式"""
        df_temp = pd.DataFrame(pd.read_hdf(h5_path, 'temperature'))
        df_humid = pd.DataFrame(pd.read_hdf(h5_path, 'humidity'))

        # 对齐索引 (两者应已在预处理中对齐)
        common_idx = df_temp.index.intersection(df_humid.index)
        common_cols = df_temp.columns.intersection(df_humid.columns)
        df_temp = df_temp.loc[common_idx, common_cols]
        df_humid = df_humid.loc[common_idx, common_cols]

        # 生成各自的 mask
        mask_temp = (~df_temp.isna()).astype('uint8').values  # (T, N)
        mask_humid = (~df_humid.isna()).astype('uint8').values  # (T, N)

        # 联合 mask: 两个通道独立 → (T, N, 2)
        mask_both = np.stack([mask_temp, mask_humid], axis=-1)  # (T, N, 2)
        # 用于 PandasDataset 的 2D mask: 取两者交集 (任一缺失则该节点该时步缺失)
        mask_2d = (mask_temp & mask_humid)

        # eval_mask 在两个通道上共享相同的传感器划分
        if self.split_file:
            known_mask = self._load_split_known_mask(mask_2d.shape[1])
            eval_mask_2d = np.tile((~known_mask).astype('uint8')[None, :],
                                   (mask_2d.shape[0], 1))
            print(f"[RoomCov Split] {self.split_file}: "
                  f"hidden={(~known_mask).sum()}/{mask_2d.shape[1]}, "
                  f"known={known_mask.sum()}/{mask_2d.shape[1]}")
        else:
            eval_mask_2d = sample_mask(mask_2d.shape, p=0., p_noise=p, mode='road')
        eval_mask_2d = (eval_mask_2d & mask_2d).astype('uint8')
        # 扩展到 (T, N, 2)
        eval_mask_both = np.stack([
            (eval_mask_2d & mask_temp).astype('uint8'),
            (eval_mask_2d & mask_humid).astype('uint8')
        ], axis=-1)
        self.eval_mask = eval_mask_2d  # 2D for PandasDataset compat

        # 填充 NaN
        if impute_nans:
            for df in [df_temp, df_humid]:
                hourly_mean = df.groupby(df.index.hour).transform('mean')
                filled = df.fillna(hourly_mean)
                filled = filled.fillna(method='ffill').fillna(method='bfill')
                filled = filled.fillna(filled.mean())
                filled = filled.fillna(0)
                df.update(filled)

        # 堆叠为 (T, N, 2) — 存储供 numpy() 返回
        data_both = np.stack([df_temp.values, df_humid.values], axis=-1)  # (T, N, 2)
        self._both_data = data_both.astype(np.float32)
        self._both_mask = mask_both
        self._both_eval_mask = eval_mask_both

        dist = np.load(os.path.join(base_path, 'dist.npy'))
        # 返回温度 df 作为 PandasDataset 的主 DataFrame (用于索引管理)
        return df_temp, dist, mask_2d

    def _load_split_known_mask(self, n_sensors):
        split = np.load(self.split_file, allow_pickle=True)
        known_mask = split["known_mask"].astype(bool)
        if known_mask.shape[0] != n_sensors:
            raise ValueError(
                f"split_file has {known_mask.shape[0]} sensors, expected {n_sensors}")
        return known_mask

    def numpy(self, return_idx=False):
        """
        返回数据数组。
        单变量: (T, N) → 经 check_dim → (T, N, 1)
        多变量: (T, N, 2) 直接返回 → check_dim 透传
        """
        if self._multivar_data is not None:
            if return_idx:
                return self._multivar_data, self.df.index
            return self._multivar_data
        return super().numpy(return_idx=return_idx)

    def splitter(self, dataset, val_len=0.1, in_sample=False, window=0):
        """
        数据集划分: 按时间顺序 70/10/20
        """
        n = len(dataset)
        test_len = int(0.2 * n)
        val_len_actual = int(0.1 * n)

        test_idxs = np.arange(n - test_len, n)
        val_idxs = np.arange(n - test_len - val_len_actual, n - test_len)
        train_idxs = np.arange(0, n - test_len - val_len_actual)

        return [train_idxs, val_idxs, test_idxs]

    def get_similarity(self, thr=0.1, include_self=False,
                       use_multi_relation=True, wall_lambda=0.5,
                       room_boost_factor=1.0, add_same_room_edges=False,
                       conn_lambda=0.0, **kwargs):
        """
        构建邻接矩阵

        Args:
            thr: 高斯核阈值
            include_self: 是否包含自环
            use_multi_relation: True 时融合距离+连接概率+房间信息 (Phase 2)
            wall_lambda: 墙体衰减系数
            room_boost_factor: 同房间增强因子
            add_same_room_edges: 是否恢复同室被截断的边

        Returns:
            adj: (N, N) 邻接矩阵
        """
        if use_multi_relation:
            adj, self._adj_components = build_multi_relation_adj(
                dist=self.dist,
                conn=self.conn,
                room=self.room,
                walls=self.walls,
                thr=thr,
                wall_lambda=wall_lambda,
                room_boost_factor=room_boost_factor,
                add_same_room_edges=add_same_room_edges,
                conn_lambda=conn_lambda,
            )
            n_edges = (adj > 0).sum()
            n_added = self._adj_components.get('same_room_edges_added', 0)
            print(f"[Multi-Relation Adj] edges={n_edges}, same-room edges added={n_added}")
        else:
            # Phase 1 baseline: 纯距离邻接
            theta = np.std(self.dist)
            adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)

        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        return adj

    @property
    def mask(self):
        if self._multivar_mask is not None:
            return self._multivar_mask
        return self._mask

    @property
    def training_mask(self):
        if self._multivar_mask is not None:
            if self._multivar_eval_mask is None:
                return self._multivar_mask
            return (self._multivar_mask & (1 - self._multivar_eval_mask))
        return self._mask if self.eval_mask is None else (self._mask & (1 - self.eval_mask))

    @property
    def n_variables(self):
        """返回变量数: 1 (单变量) 或 2 (温度+湿度)"""
        return 2 if self._multivar_data is not None else 1
