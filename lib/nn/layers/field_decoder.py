"""
Neural Field Decoder — 从传感器隐状态预测任意空间位置的温度

架构:
    Query (x,y) → CoordEncoder → q
    Sensor hidden → Cross-Attention(q, k=sensor, v=sensor) → 聚合
    Wall mask → attention衰减 (跨墙传感器权重降低)
    [q; attn_out; phys] → MLP → temperature

训练策略:
    - 利用现有kriging任务: 留出传感器位置作为query points
    - supervision = 留出传感器的真实温度
    - 无需额外数据或标注
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FieldDecoder(nn.Module):
    """
    从KITS encoder隐状态预测任意空间位置的温度.

    支持两种模式:
    1. 训练模式: query = 留出传感器坐标, supervision = 真实温度
    2. 推断模式: query = 任意dense grid坐标
    """

    def __init__(self, d_hidden, n_heads=4, dropout=0.1,
                 use_wall_mask=True, coord_dim=2):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.use_wall_mask = use_wall_mask

        # 坐标编码: (x, y) → d_hidden
        # 使用 sinusoidal position encoding 增强空间分辨率
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
        )

        # 传感器坐标编码 (用于relative position)
        self.sensor_coord_proj = nn.Linear(coord_dim, d_hidden)

        # 跨注意力: query=grid点, key/value=传感器隐状态
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,  # (B, Q, D) format
        )

        # 输出MLP: [attn_out + coord_embed] → temperature
        self.output_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # 墙体衰减参数 (可学习)
        if use_wall_mask:
            # log-scale wall penalty: attention *= exp(-wall_penalty * n_walls)
            self.wall_penalty = nn.Parameter(torch.tensor(1.0))

    def forward(self, encoder_hidden, sensor_coords, query_coords,
                wall_crossings=None, time_avg=True):
        """
        Args:
            encoder_hidden: (B, D, N, S) KITS encoder输出
            sensor_coords: (N, 2) 传感器坐标
            query_coords: (Q, 2) 查询点坐标
            wall_crossings: (Q, N) 每个query点到每个传感器的穿墙数量
                           如果None, 不使用墙体mask
            time_avg: 是否对时间步取平均 (True用于空间场重建)

        Returns:
            predictions: (B, Q, 1) 预测温度
        """
        B, D, N, S = encoder_hidden.shape
        Q = query_coords.shape[0]
        device = encoder_hidden.device

        # 1. 时间聚合: (B, D, N, S) → (B, N, D)
        if time_avg:
            sensor_features = encoder_hidden.mean(dim=-1)  # (B, D, N)
            sensor_features = sensor_features.permute(0, 2, 1)  # (B, N, D)
        else:
            # 取最后时间步
            sensor_features = encoder_hidden[:, :, :, -1]  # (B, D, N)
            sensor_features = sensor_features.permute(0, 2, 1)  # (B, N, D)

        # 2. 加入传感器坐标信息到key/value
        sensor_coord_embed = self.sensor_coord_proj(
            torch.tensor(sensor_coords, dtype=torch.float32, device=device))  # (N, D)
        sensor_features = sensor_features + sensor_coord_embed.unsqueeze(0)  # (B, N, D)

        # 3. 编码查询坐标
        query_embed = self.coord_encoder(
            torch.tensor(query_coords, dtype=torch.float32, device=device))  # (Q, D)
        query_embed = query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, Q, D)

        # 4. 构建墙体注意力mask
        attn_mask = None
        if self.use_wall_mask and wall_crossings is not None:
            # wall_crossings: (Q, N) → attention penalty
            # 更多墙 → 更低的注意力权重
            wc = torch.tensor(wall_crossings, dtype=torch.float32, device=device)
            penalty = self.wall_penalty.abs()
            # MHA expects: (B*n_heads, Q, N) or (Q, N) additive mask
            # 使用additive mask: 负值降低attention score
            attn_mask = -penalty * wc  # (Q, N), 负值
            # 扩展到多头: MHA broadcast automatically if 2D

        # 5. 跨注意力
        attn_out, attn_weights = self.cross_attn(
            query=query_embed,      # (B, Q, D)
            key=sensor_features,    # (B, N, D)
            value=sensor_features,  # (B, N, D)
            attn_mask=attn_mask,    # (Q, N) or None
        )
        # attn_out: (B, Q, D)

        # 6. 拼接并输出
        combined = torch.cat([attn_out, query_embed], dim=-1)  # (B, Q, 2D)
        predictions = self.output_mlp(combined)  # (B, Q, 1)

        return predictions, attn_weights

    def compute_field_loss(self, predictions, targets, mask=None):
        """
        计算场重建loss.

        Args:
            predictions: (B, Q, 1) 预测值
            targets: (B, Q, 1) 真实值
            mask: (B, Q, 1) 有效位置mask, 可选

        Returns:
            loss: scalar
        """
        if mask is not None:
            diff = (predictions - targets).abs() * mask
            loss = diff.sum() / (mask.sum() + 1e-8)
        else:
            loss = F.l1_loss(predictions, targets)
        return loss
