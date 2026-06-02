"""
Physics-Informed Loss: 图上离散热扩散方程约束

连续形式: ∂T/∂t = α · ∇²T
图上离散: dT/dt ≈ α · L · T,  L = D - A (图拉普拉斯矩阵)

损失: L_physics = mean(||dT/dt - α · L · T||²)

墙体感知: 邻接矩阵 A 已包含墙体衰减 (exp(-λ·num_walls)),
          因此拉普拉斯算子自然地在跨墙传感器之间减弱扩散。
"""

import torch
from torch import nn


class PhysicsLoss(nn.Module):
    """
    基于热扩散方程的物理约束损失

    计算模型预测温度场与热扩散方程的偏差:
      residual = ∂T/∂t - α · Δ_G · T
      L_physics = mean(residual²)
    """

    def __init__(self, alpha=0.01, normalize=True):
        """
        Args:
            alpha: 热扩散系数 (可学习)
            normalize: 是否对拉普拉斯矩阵归一化
        """
        super().__init__()
        # 可学习的热扩散系数
        self.log_alpha = nn.Parameter(torch.tensor(float(alpha)).log())
        self.normalize = normalize

    @property
    def alpha(self):
        """确保 alpha > 0"""
        return self.log_alpha.exp()

    def compute_graph_laplacian(self, adj):
        """
        计算图拉普拉斯矩阵: L = D - A

        Args:
            adj: (N, N) 邻接矩阵

        Returns:
            L: (N, N) 拉普拉斯矩阵
        """
        D = torch.diag(adj.sum(dim=1))
        L = D - adj

        if self.normalize:
            # 归一化拉普拉斯: L_norm = D^{-1/2} L D^{-1/2}
            d_inv_sqrt = torch.diag(1.0 / (adj.sum(dim=1).sqrt() + 1e-8))
            L = d_inv_sqrt @ L @ d_inv_sqrt

        return L

    def forward(self, T_pred, adj, mask=None):
        """
        计算物理损失

        Args:
            T_pred: (B, S, N, C) 模型预测场 (C=1温度 或 C=2温度+湿度)
            adj: (N, N) 邻接矩阵 (已含墙体衰减)
            mask: (B, S, N, C) 可选, 仅在有观测的时间步和节点上计算

        Returns:
            loss: 标量, 热扩散方程残差的均方值
        """
        B, S, N, C = T_pred.shape

        if S < 2:
            return torch.tensor(0.0, device=T_pred.device, requires_grad=True)

        L = self.compute_graph_laplacian(adj)  # (N, N)

        total_loss = torch.tensor(0.0, device=T_pred.device, requires_grad=True)
        for ch in range(C):
            # 时间导数: ∂T/∂t ≈ T(t+1) - T(t)
            dT_dt = T_pred[:, 1:, :, ch:ch+1] - T_pred[:, :-1, :, ch:ch+1]  # (B, S-1, N, 1)

            # 空间拉普拉斯: Δ_G · T = L · T
            T_for_lap = T_pred[:, :-1, :, ch]  # (B, S-1, N)
            lap_T = torch.matmul(T_for_lap, L.T)  # (B, S-1, N)
            lap_T = lap_T.unsqueeze(-1)  # (B, S-1, N, 1)

            residual = dT_dt - self.alpha * lap_T  # (B, S-1, N, 1)

            if mask is not None:
                mask_ch = mask[:, :, :, ch:ch+1]  # (B, S, N, 1)
                mask_valid = mask_ch[:, :-1, :, :] * mask_ch[:, 1:, :, :]
                if mask_valid.sum() == 0:
                    continue
                total_loss = total_loss + (residual ** 2 * mask_valid).sum() / mask_valid.sum()
            else:
                total_loss = total_loss + (residual ** 2).mean()

        return total_loss / max(C, 1)


class SpatialSmoothLoss(nn.Module):
    """
    空间平滑正则损失

    同一房间内的传感器温度应该相对平滑 (梯度小),
    跨房间的温度差异可以更大。

    L_smooth = mean_same_room(|T_i - T_j|²) - λ · mean_cross_room(|T_i - T_j|²)
    简化为: L_smooth = sum_{(i,j) in same_room} A_ij · (T_i - T_j)²
    """

    def __init__(self):
        super().__init__()

    def forward(self, T_pred, adj_room):
        """
        Args:
            T_pred: (B, S, N, C) C=1 or 2
            adj_room: (N, N) 同一房间邻接矩阵

        Returns:
            loss: 标量
        """
        B, S, N, C = T_pred.shape

        room_mask = adj_room.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        n_pairs = room_mask.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=T_pred.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=T_pred.device, requires_grad=True)
        for ch in range(C):
            T = T_pred[:, :, :, ch]  # (B, S, N)
            T_diff = T.unsqueeze(-1) - T.unsqueeze(-2)  # (B, S, N, N)
            T_diff_sq = T_diff ** 2
            weighted = T_diff_sq * room_mask
            total_loss = total_loss + weighted.sum() / (n_pairs * B * S)

        return total_loss / max(C, 1)


class ConnSpatialLoss(nn.Module):
    """
    A1: conn 加权各向异性空间一致性损失

    conn 高的传感器对 → 预测温度应一致 (同室, 无墙阻隔)
    conn 低的传感器对 → 允许温度跳变 (跨墙)

    L = Σ_{i,j} conn(i,j) × (T̂_i - T̂_j)² / (dist(i,j) + ε)

    与 SpatialSmoothLoss 的区别:
    - SpatialSmoothLoss 用二值 room 矩阵 (同室=1, 跨室=0)
    - ConnSpatialLoss 用连续 conn 值, 保留了室内远近差异和跨墙强弱差异
    """

    def __init__(self):
        super().__init__()

    def forward(self, T_pred, conn, dist=None):
        """
        Args:
            T_pred: (B, S, N, C)
            conn: (N, N) 连接概率矩阵 (0-1连续值)
            dist: (N, N) 距离矩阵, 用于归一化 (可选)

        Returns:
            loss: 标量
        """
        B, S, N, C = T_pred.shape

        # conn 权重矩阵, 去除自环
        w = conn.clone()
        w.fill_diagonal_(0)

        if dist is not None:
            # 距离归一化: 近邻权重更高
            d = dist.clone()
            d.fill_diagonal_(1)  # 避免除零
            w = w / (d + 1e-6)

        w = w.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        w_sum = w.sum()
        if w_sum == 0:
            return torch.tensor(0.0, device=T_pred.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=T_pred.device, requires_grad=True)
        for ch in range(C):
            T = T_pred[:, :, :, ch]  # (B, S, N)
            T_diff_sq = (T.unsqueeze(-1) - T.unsqueeze(-2)) ** 2  # (B, S, N, N)
            total_loss = total_loss + (T_diff_sq * w).sum() / (w_sum * B * S)

        return total_loss / max(C, 1)


class ConnGradientLoss(nn.Module):
    """
    A2: conn 加权时间梯度一致性损失

    同室传感器的温度变化趋势 (升/降温速率) 应一致,
    即使绝对温度因局部热源不同。

    ∇T_i(t) = T̂_i(t) - T̂_i(t-1)
    L = Σ_{i,j} conn(i,j) × (∇T_i(t) - ∇T_j(t))²

    物理含义: 同一房间受相同 HVAC/环境驱动, 温度变化方向和速率应一致
    """

    def __init__(self):
        super().__init__()

    def forward(self, T_pred, conn):
        """
        Args:
            T_pred: (B, S, N, C)  S >= 2
            conn: (N, N) 连接概率矩阵

        Returns:
            loss: 标量
        """
        B, S, N, C = T_pred.shape
        if S < 2:
            return torch.tensor(0.0, device=T_pred.device, requires_grad=True)

        # conn 权重, 去除自环
        w = conn.clone()
        w.fill_diagonal_(0)
        w = w.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        w_sum = w.sum()
        if w_sum == 0:
            return torch.tensor(0.0, device=T_pred.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=T_pred.device, requires_grad=True)
        for ch in range(C):
            # 时间梯度: ∂T/∂t
            grad_T = T_pred[:, 1:, :, ch] - T_pred[:, :-1, :, ch]  # (B, S-1, N)
            # 梯度差异
            grad_diff_sq = (grad_T.unsqueeze(-1) - grad_T.unsqueeze(-2)) ** 2  # (B, S-1, N, N)
            total_loss = total_loss + (grad_diff_sq * w).sum() / (w_sum * B * (S - 1))

        return total_loss / max(C, 1)
