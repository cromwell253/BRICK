"""
Multi-Scale Temporal Pyramid (Phase 7)

替换 KITS 的三层同质 F.unfold(kernel=3) 为多尺度时序编码:
  Layer 1: kernel=3  (局部, 1.5小时感受野)
  Layer 2: kernel=5  (中程, 2.5小时感受野)
  Layer 3: kernel=7  (长程, 3.5小时感受野)

创新点:
  不同层使用不同时序感受野, 分别捕获:
  - 快速温度波动 (空调开关, 人员进出)
  - 小时级周期 (工作/休息交替)
  - 趋势性变化 (日间升温/夜间降温)
"""

import torch
import torch.nn.functional as F


# Multi-scale kernel sizes for the 3 GCN layers
DEFAULT_KERNELS = [3, 5, 7]


def multi_scale_unfold(x, kernel_size, n_nodes):
    """
    Apply F.unfold with specified kernel_size along time dimension.

    Args:
        x: (B, D, S, N) tensor - note S is time, N is nodes
        kernel_size: temporal kernel size
        n_nodes: number of nodes N

    Returns:
        out: (B, kernel_size * D, N', S) tensor
    """
    pad = kernel_size // 2
    out = F.unfold(x, kernel_size=(kernel_size, n_nodes),
                   padding=(pad, 0), stride=(1, 1))
    d = x.size(1)
    s = x.size(2)
    out = out.reshape(x.size(0), kernel_size * d, -1, s)  # B, k*D, N', S
    return out
