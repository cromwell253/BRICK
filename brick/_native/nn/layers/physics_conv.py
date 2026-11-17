pass
import torch
from torch import nn
from ... import epsilon

class PhysicsGuidedConv(nn.Module):
    pass

    def __init__(self, c_in, c_out, support_len=2, order=1, include_self=False):
        pass
        super().__init__()
        self.include_self = include_self
        self.order = order
        self.c_out = c_out
        data_c_in = (order * support_len + (1 if include_self else 0)) * c_in
        self.data_mlp = nn.Conv2d(data_c_in, c_out, kernel_size=1)
        self.phys_proj = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.log_alpha = nn.Parameter(torch.tensor(0.01).log())
        self.log_dt = nn.Parameter(torch.tensor(1.0).log())
        self.gate_fc = nn.Sequential(nn.Conv2d(2 * c_out, c_out, kernel_size=1), nn.Sigmoid())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def dt(self):
        return self.log_dt.exp()

    @staticmethod
    def compute_support(adj, device=None):
        pass
        if device is not None:
            adj = adj.to(device)
        adj_bwd = adj.T
        adj_fwd = adj / (adj.sum(1, keepdims=True) + epsilon)
        adj_bwd = adj_bwd / (adj_bwd.sum(1, keepdims=True) + epsilon)
        return [adj_fwd, adj_bwd]

    def _compute_data_branch(self, x, support):
        pass
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
        pass
        h = self.phys_proj(x)
        a_fwd = support[0]
        ah = torch.einsum('ncvl,wv->ncwl', (h, a_fwd)).contiguous()
        coeff = self.alpha * self.dt
        h_phys = (1.0 - coeff) * h + coeff * ah
        return h_phys

    def forward(self, x, support):
        pass
        if x.dim() < 4:
            squeeze = True
            x = torch.unsqueeze(x, -1)
        else:
            squeeze = False
        h_data = self._compute_data_branch(x, support)
        h_phys = self._compute_physics_branch(x, support)
        gate = self.gate_fc(torch.cat([h_data, h_phys], dim=1))
        out = gate * h_phys + (1 - gate) * h_data
        if squeeze:
            out = out.squeeze(-1)
        return out
