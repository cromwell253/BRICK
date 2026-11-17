pass
import torch
from torch import nn
from einops import rearrange

class WallAwareAttention(nn.Module):
    pass

    def __init__(self, d_hidden, n_heads=4, dropout=0.1):
        pass
        super().__init__()
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = d_hidden // n_heads
        assert d_hidden % n_heads == 0, f'd_hidden ({d_hidden}) must be divisible by n_heads ({n_heads})'
        self.W_q = nn.Linear(d_hidden, d_hidden)
        self.W_k = nn.Linear(d_hidden, d_hidden)
        self.W_v = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)
        self.log_lambda = nn.Parameter(torch.tensor(0.5).log())
        self.gate = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.Sigmoid())
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** (-0.5)

    @property
    def wall_lambda(self):
        return self.log_lambda.exp()

    def forward(self, feat, mask, walls=None):
        pass
        (BS, N, D) = feat.shape
        Q = self.W_q(feat)
        K = self.W_k(feat)
        V = self.W_v(feat)
        Q = Q.view(BS, N, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(BS, N, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(BS, N, self.n_heads, self.d_head).transpose(1, 2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if walls is not None:
            wall_bias = -self.wall_lambda * walls
            wall_bias = wall_bias.unsqueeze(0).unsqueeze(0)
            attn = attn + wall_bias
        if mask.dim() == 3 and mask.size(-1) > 1:
            mask_flat = mask.any(dim=-1).float()
        else:
            mask_flat = mask.squeeze(-1).float()
        obs_mask = mask_flat.unsqueeze(-1)
        unobs_mask = 1.0 - obs_mask
        cross_mask = unobs_mask @ obs_mask.transpose(-2, -1) + obs_mask @ unobs_mask.transpose(-2, -1)
        same_mask = obs_mask @ obs_mask.transpose(-2, -1) + unobs_mask @ unobs_mask.transpose(-2, -1)
        attn_mask = (cross_mask + same_mask).clamp(0, 1)
        attn_mask = attn_mask.unsqueeze(1)
        attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(BS, N, D)
        out = self.W_o(out)
        gate = self.gate(torch.cat([feat, out], dim=-1))
        out = gate * out + (1 - gate) * feat
        return out
