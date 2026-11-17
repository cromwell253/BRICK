pass
import torch
from torch import nn

class TemporalGRU(nn.Module):
    pass

    def __init__(self, d_hidden, num_layers=1, dropout=0.0, bidirectional=True):
        pass
        super().__init__()
        self.d_hidden = d_hidden
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(input_size=d_hidden, hidden_size=d_hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=bidirectional)
        if bidirectional:
            self.proj = nn.Linear(2 * d_hidden, d_hidden)
        else:
            self.proj = nn.Identity()
        self.gate = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.Sigmoid())

    def forward(self, x):
        pass
        (B, D, N, S) = x.shape
        x_in = x.permute(0, 2, 3, 1).reshape(B * N, S, D)
        (gru_out, _) = self.gru(x_in)
        gru_out = self.proj(gru_out)
        gate = self.gate(gru_out)
        out = gate * gru_out + (1 - gate) * x_in
        out = out.reshape(B, N, S, D).permute(0, 3, 1, 2)
        return out
