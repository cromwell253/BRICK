"""
Physics-Guided Graph Convolution (Phase 6)

替换通用的 SpatialConvOrderK, 将热传导 PDE 离散化嵌入图卷积核。

原始 GCN:  H' = sigma(MLP([A·H; A^T·H]))   (纯数据驱动)
改进 PGC:  H_phys = (I + alpha*dt*L) @ H      (热扩散一步)
           H_data = sigma(MLP([A·H; A^T·H]))  (数据驱动)
           gate = sigma(Linear([H_phys; H_data]))
           H' = gate * H_phys + (1-gate) * H_data

创新点:
  物理过程不再仅作为损失约束, 而是直接嵌入卷积算子,
  使每一层图卷积都隐含热扩散方程的先验。
"""

import torch
from torch import nn

from ... import epsilon


class PhysicsGuidedConv(nn.Module):
    """
    物理引导的图卷积层

    将图拉普拉斯热扩散离散化作为一条物理支路,
    与数据驱动的图卷积支路进行门控融合。
    """

    def __init__(self, c_in, c_out, support_len=2, order=1, include_self=False):
        """
        Args:
            c_in: 输入通道数 (通常 = t_dim * d_hidden)
            c_out: 输出通道数 (= d_hidden)
            support_len: 支持矩阵数量 (fwd + bwd = 2)
            order: 图扩散阶数
            include_self: 是否包含自环
        """
        super().__init__()
        self.include_self = include_self
        self.order = order
        self.c_out = c_out

        # Data-driven branch (same as SpatialConvOrderK)
        data_c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.data_mlp = nn.Conv2d(data_c_in, c_out, kernel_size=1)

        # Physics branch: project c_in to c_out, then apply diffusion
        self.phys_proj = nn.Conv2d(c_in, c_out, kernel_size=1)

        # Learnable diffusion parameters (log space for positivity)
        self.log_alpha = nn.Parameter(torch.tensor(0.01).log())
        self.log_dt = nn.Parameter(torch.tensor(1.0).log())

        # Gating: fuse physics and data branches
        self.gate_fc = nn.Sequential(
            nn.Conv2d(2 * c_out, c_out, kernel_size=1),
            nn.Sigmoid()
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def dt(self):
        return self.log_dt.exp()

    @staticmethod
    def compute_support(adj, device=None):
        """Same as SpatialConvOrderK.compute_support"""
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + epsilon)
        return [adj_fwd, adj_bwd]

    def _compute_data_branch(self, x, support):
        """Data-driven graph convolution (same logic as SpatialConvOrderK)"""
        out = [x] if self.include_self else []
        if not isinstance(support, list):
            support = [support]
        for a in support:
            x1 = torch.einsum('ncvl,wv->ncwl', (x, a)).contiguous()
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = torch.einsum('ncvl,wv->ncwl', (x1, a)).contiguous()
                out.append(x2)
                x1 = x2
        out = torch.cat(out, dim=1)
        return self.data_mlp(out)

    def _compute_physics_branch(self, x, support):
        """
        Physics-guided diffusion step:
          H_phys = (I + alpha * dt * L_sym) @ H

        where L_sym = I - (D^{-1/2} A D^{-1/2}) is the symmetric normalized Laplacian
        Using the row-normalized support: L_rw = I - A_rw ≈ L_sym for undirected graphs
        """
        # Project to c_out channels first
        h = self.phys_proj(x)  # (B, c_out, N, S)

        # Use forward support (row-normalized adj) for diffusion
        a_fwd = support[0]  # (N, N) row-normalized

        # Diffusion: A_rw @ H (smoothing step)
        # h shape: (B, C, N, S)
        ah = torch.einsum('ncvl,wv->ncwl', (h, a_fwd)).contiguous()

        # Heat diffusion: H + alpha*dt*(A@H - H) = H + alpha*dt*(-L@H)
        # = (1 - alpha*dt)*H + alpha*dt*(A@H)
        coeff = self.alpha * self.dt
        h_phys = (1.0 - coeff) * h + coeff * ah

        return h_phys

    def forward(self, x, support):
        """
        Args:
            x: (B, C_in, N, S) input features
            support: list of (N, N) support matrices

        Returns:
            out: (B, C_out, N, S)
        """
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False

        # Two branches
        h_data = self._compute_data_branch(x, support)    # (B, c_out, N, S)
        h_phys = self._compute_physics_branch(x, support)  # (B, c_out, N, S)

        # Gated fusion
        gate = self.gate_fc(torch.cat([h_data, h_phys], dim=1))
        out = gate * h_phys + (1 - gate) * h_data

        if squeeze:
            out = out.squeeze(-1)
        return out
