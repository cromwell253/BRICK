"""
Temporal Enhancement Module for PI-KITS

在空间GCN之后增加显式时序建模，捕获长程时间依赖。

当前KITS仅使用 F.unfold(kernel=3) 做局部时序聚合，
有效感受野仅 ~7步。GRU模块可建模整个序列的时序依赖。

插入位置: Hard Transfer 之后, gcn_3 之前
数据流: [B, D, N, S] → permute → [B*N, S, D] → GRU → reshape → [B, D, N, S]
"""

import torch
from torch import nn


class TemporalGRU(nn.Module):
    """
    GRU 时序增强模块

    对每个节点独立地沿时间维度做 GRU 编码，
    捕获长程时间依赖关系。
    """

    def __init__(self, d_hidden, num_layers=1, dropout=0.0, bidirectional=True):
        """
        Args:
            d_hidden: 隐藏层维度 (应与 KITS 的 d_hidden 一致)
            num_layers: GRU 层数
            dropout: GRU 层间 dropout (仅在 num_layers > 1 时有效)
            bidirectional: 是否使用双向 GRU (场重建为插值任务，非因果预测)
        """
        super().__init__()
        self.d_hidden = d_hidden
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=d_hidden,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # 如果双向，需要投影回 d_hidden
        if bidirectional:
            self.proj = nn.Linear(2 * d_hidden, d_hidden)
        else:
            self.proj = nn.Identity()

        # 残差连接的门控
        self.gate = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, D, N, S) — D=d_hidden, S=时间步, N=节点数

        Returns:
            out: (B, D, N, S) — 同维输出 (残差连接)
        """
        B, D, N, S = x.shape

        # (B, D, N, S) → (B, N, S, D) → (B*N, S, D)
        x_in = x.permute(0, 2, 3, 1).reshape(B * N, S, D)

        # GRU 前向
        gru_out, _ = self.gru(x_in)  # (B*N, S, D*num_directions)

        # 投影回 d_hidden
        gru_out = self.proj(gru_out)  # (B*N, S, D)

        # 门控残差连接: out = gate * gru_out + (1 - gate) * x_in
        gate = self.gate(gru_out)  # (B*N, S, D)
        out = gate * gru_out + (1 - gate) * x_in

        # (B*N, S, D) → (B, N, S, D) → (B, D, N, S)
        out = out.reshape(B, N, S, D).permute(0, 3, 1, 2)

        return out
