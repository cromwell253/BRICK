"""
Wall-Aware Cross-Attention Module (Phase 5)

替换 KITS 的 Hard Transfer 机制:
  原始: 余弦相似度 -> 硬匹配(1对1) -> 特征复制
  改进: Q,K,V变换 -> 墙体偏置注意力(多对多) -> 加权融合

创新点:
  将建筑结构先验 (墙体数量) 嵌入注意力偏置矩阵,
  使信息传递在跨墙方向自动衰减, 同房间方向增强。

  Attention(Q,K,V) = softmax(QK^T / sqrt(d) + wall_bias) @ V
  wall_bias[i,j] = -lambda * walls[i,j]  (lambda 可学习)
"""

import torch
from torch import nn
from einops import rearrange


class WallAwareAttention(nn.Module):
    """
    墙体感知的交叉注意力模块

    核心机制:
    1. Q, K, V 线性变换
    2. 注意力分数 = QK^T/sqrt(d) + wall_bias
       wall_bias[i,j] = -exp(log_lambda) * walls[i,j]
    3. 对观测/未观测节点分别做交叉注意力
    4. 门控残差连接
    """

    def __init__(self, d_hidden, n_heads=4, dropout=0.1):
        """
        Args:
            d_hidden: 隐藏层维度
            n_heads: 注意力头数
            dropout: 注意力 dropout
        """
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        assert d_hidden % n_heads == 0, \
            f"d_hidden ({d_hidden}) must be divisible by n_heads ({n_heads})"

        # Q, K, V projections
        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_k = nn.Linear(d_hidden, d_hidden)
        self.W_v = nn.Linear(d_hidden, d_hidden)

        # Output projection
        self.W_o = nn.Linear(d_hidden, d_hidden)

        # Learnable wall penalty coefficient (log space for positivity)
        self.log_lambda = nn.Parameter(torch.tensor(0.5).log())

        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.Sigmoid()
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    @property
    def wall_lambda(self):
        return self.log_lambda.exp()

    def forward(self, feat, mask, walls=None):
        """
        Args:
            feat: (B*S, N, D) 节点特征
            mask: (B*S, N, 1) 观测掩码 (1=observed, 0=unobserved)
            walls: (N, N) 墙体数量矩阵 (可选, None时退化为标准注意力)

        Returns:
            out: (B*S, N, D) 融合后特征
        """
        BS, N, D = feat.shape

        # Q, K, V
        Q = self.W_q(feat)  # (BS, N, D)
        K = self.W_k(feat)  # (BS, N, D)
        V = self.W_v(feat)  # (BS, N, D)

        # Multi-head reshape: (BS, N, D) -> (BS, H, N, d_head)
        Q = Q.view(BS, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(BS, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(BS, N, self.n_heads, self.d_head).transpose(1, 2)

        # Attention scores: (BS, H, N, N)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Wall-aware bias: penalize cross-wall attention
        if walls is not None:
            wall_bias = -self.wall_lambda * walls  # (N, N)
            wall_bias = wall_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            attn = attn + wall_bias

        # Mask-aware attention: obs->unobs and unobs->obs
        # Create cross-attention mask:
        #   For unobserved nodes: attend to observed nodes
        #   For observed nodes: attend to unobserved nodes (for cycle consistency)
        # Reduce multi-channel mask to node-level: observed if any channel observed
        if mask.dim() == 3 and mask.size(-1) > 1:
            mask_flat = mask.any(dim=-1).float()  # (BS, N)
        else:
            mask_flat = mask.squeeze(-1).float()  # (BS, N), ensure float
        obs_mask = mask_flat.unsqueeze(-1)  # (BS, N, 1) - which nodes are observed
        unobs_mask = 1.0 - obs_mask  # (BS, N, 1) - which nodes are unobserved

        # Cross attention targets:
        # unobs nodes attend to obs nodes: mask_target[unobs_i, obs_j] = 1
        # obs nodes attend to unobs nodes: mask_target[obs_i, unobs_j] = 1
        cross_mask = (unobs_mask @ obs_mask.transpose(-2, -1) +
                      obs_mask @ unobs_mask.transpose(-2, -1))  # (BS, N, N)
        # Also allow self-attention within same category
        same_mask = (obs_mask @ obs_mask.transpose(-2, -1) +
                     unobs_mask @ unobs_mask.transpose(-2, -1))

        # Final mask: both cross and same category attention
        attn_mask = (cross_mask + same_mask).clamp(0, 1)
        attn_mask = attn_mask.unsqueeze(1)  # (BS, 1, N, N)

        # Apply mask (set non-attended positions to -inf)
        attn = attn.masked_fill(attn_mask == 0, float('-inf'))

        # Softmax + dropout
        attn = torch.softmax(attn, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)  # handle all-masked rows
        attn = self.attn_dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, V)  # (BS, H, N, d_head)
        out = out.transpose(1, 2).contiguous().view(BS, N, D)  # (BS, N, D)
        out = self.W_o(out)  # (BS, N, D)

        # Gated residual connection
        gate = self.gate(torch.cat([feat, out], dim=-1))
        out = gate * out + (1 - gate) * feat

        return out
