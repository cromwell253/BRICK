"""
Multi-Relation Adjacency Matrix Construction

将距离、连接概率、房间关系三种邻接矩阵融合为一个墙体感知的邻接矩阵。

融合策略:
  1. 以距离邻接 A_dist 为骨架
  2. 同房间节点对增强 2.0x (room_boost)
  3. 同室补边: 同一房间但被距离阈值截断的节点对，用最小权重恢复连接
  4. 墙体衰减: wall_decay[i,j] = exp(-λ · num_walls[i,j])
  5. 阈值截断保持稀疏性
"""

import numpy as np
import torch
from torch import nn


def build_multi_relation_adj(dist, conn, room, walls, thr=0.1, wall_lambda=0.5,
                             room_boost_factor=1.0, add_same_room_edges=False,
                             conn_lambda=0.0):
    """
    构建多关系融合邻接矩阵 (NumPy, 用于初始化)

    融合策略:
      1. 以距离邻接 A_dist 为骨架
      2. 连接概率软调制: A_dist × (1 - λ_c + λ_c × A_conn_norm)
         conn隐式编码墙体: 同室conn≈0.37, 1墙≈0.14, 2墙≈0.06, 3墙≈0.04
      3. 同房间节点对增强 (room_boost)
      4. (可选) 墙体衰减: wall_decay = exp(-λ_w · num_walls)
      5. 阈值截断保持稀疏性

    Args:
        dist: (N, N) 欧氏距离矩阵
        conn: (N, N) 对称化连接概率矩阵
        room: (N, N) 同一房间矩阵 (0/1)
        walls: (N, N) 墙体数量矩阵
        thr: 高斯核阈值
        wall_lambda: 墙体衰减系数 (0=无墙体衰减)
        room_boost_factor: 同房间增强因子 (默认1.0 → 2.0x boost)
        add_same_room_edges: 是否恢复同室被截断的边
        conn_lambda: 连接概率调制强度 (0=不用conn, 1=完全调制)

    Returns:
        A_fused: (N, N) 融合邻接矩阵
        components: dict 包含各分量矩阵 (用于可视化/调试)
    """

    # ---- 关系1: 距离邻接 A_dist (骨架) ----
    theta = np.std(dist)
    A_dist = np.exp(-np.square(dist / theta))
    A_dist[A_dist < thr] = 0.0
    np.fill_diagonal(A_dist, 0.0)

    # ---- 关系2: 连接概率邻接 A_conn ----
    A_conn = conn.copy()
    conn_max = A_conn.max()
    if conn_max > 0:
        A_conn = A_conn / conn_max  # 归一化到 [0, 1]
    np.fill_diagonal(A_conn, 0.0)

    # ---- 关系3: 同一房间邻接 A_room ----
    A_room = room.copy()
    np.fill_diagonal(A_room, 0.0)

    # ---- 墙体衰减因子 ----
    wall_decay = np.exp(-wall_lambda * walls)
    np.fill_diagonal(wall_decay, 1.0)

    # ---- 融合 ----
    # 1) 连接概率软调制: 高conn(同室)→权重不变, 低conn(跨墙)→权重降低
    #    conn_factor ∈ [(1-λ_c), 1.0], 不会把任何边完全杀死
    if conn_lambda > 0:
        conn_factor = (1 - conn_lambda) + conn_lambda * A_conn
        A_fused = A_dist * conn_factor
        print(f"[Multi-Relation] conn_lambda={conn_lambda:.2f}, "
              f"conn modulation range: [{conn_factor[A_dist > 0].min():.3f}, "
              f"{conn_factor[A_dist > 0].max():.3f}]")
    else:
        A_fused = A_dist.copy()

    # 2) 同房间节点对的距离权重增强
    if room_boost_factor > 0:
        room_boost = 1.0 + room_boost_factor * A_room
        A_fused = A_fused * room_boost

    # 3) 可选: 同室补边
    n_added = 0
    if add_same_room_edges:
        room_boost = 1.0 + room_boost_factor * A_room
        same_room_missing = (A_room > 0) & (A_fused < thr)
        np.fill_diagonal(same_room_missing, False)
        A_fused[same_room_missing] = np.maximum(
            A_dist[same_room_missing] * room_boost[same_room_missing],
            thr * 1.5
        )
        n_added = int(same_room_missing.sum())

    # 4) 施加墙体穿越衰减
    if wall_lambda > 0:
        A_fused = A_fused * wall_decay

    # 保持稀疏性: 仍使用阈值截断
    A_fused[A_fused < thr] = 0.0

    # ---- 移除自环 ----
    np.fill_diagonal(A_fused, 0.0)

    components = {
        'A_dist': A_dist,
        'A_conn': A_conn,
        'A_room': A_room,
        'wall_decay': wall_decay,
        'conn_lambda': conn_lambda,
        'same_room_edges_added': n_added,
    }

    return A_fused, components


class LearnableAdjFusion(nn.Module):
    """
    可学习的多关系邻接矩阵融合模块

    在训练过程中学习三种关系的最优权重组合。
    将此模块注册到 KITS 模型中，使 α, β, γ 可梯度更新。
    """

    def __init__(self, A_dist, A_conn, A_room, wall_decay, init_weights=None):
        """
        Args:
            A_dist: (N, N) np.ndarray 距离邻接矩阵
            A_conn: (N, N) np.ndarray 连接概率邻接矩阵
            A_room: (N, N) np.ndarray 同一房间邻接矩阵
            wall_decay: (N, N) np.ndarray 墙体衰减因子
            init_weights: (3,) 初始权重, 默认等权
        """
        super().__init__()

        # 注册为 buffer (不可学习, 但随模型移动设备)
        self.register_buffer('A_dist', torch.tensor(A_dist, dtype=torch.float32))
        self.register_buffer('A_conn', torch.tensor(A_conn, dtype=torch.float32))
        self.register_buffer('A_room', torch.tensor(A_room, dtype=torch.float32))
        self.register_buffer('wall_decay', torch.tensor(wall_decay, dtype=torch.float32))

        # 可学习的融合权重 (logit 空间, 通过 softmax 归一化)
        if init_weights is None:
            init_weights = [1.0, 1.0, 1.0]
        self.relation_logits = nn.Parameter(
            torch.tensor(init_weights, dtype=torch.float32)
        )

    def forward(self):
        """
        Returns:
            A_fused: (N, N) 融合后的邻接矩阵
        """
        weights = torch.softmax(self.relation_logits, dim=0)

        A_fused = (weights[0] * self.A_dist +
                   weights[1] * self.A_conn +
                   weights[2] * self.A_room)

        # 应用墙体衰减
        A_fused = A_fused * self.wall_decay

        return A_fused

    def get_weights(self):
        """返回当前融合权重 (用于监控/日志)"""
        with torch.no_grad():
            w = torch.softmax(self.relation_logits, dim=0)
        return w.cpu().numpy()
