"""
ForecastDecoder: GRU-based autoregressive decoder for temporal forecasting.

Takes encoder hidden states from the imputation backbone and predicts
H future timesteps through step-by-step GRU decoding with spatial GCN
refinement at each step.

Phase 11 addition to PI-KITS.
"""

import torch
import torch.nn as nn
from .spatial_conv import SpatialConvOrderK


class ForecastDecoder(nn.Module):
    def __init__(self, d_hidden, d_out, forecast_horizon,
                 support_len=2, use_uncertainty=False):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.forecast_horizon = forecast_horizon
        self.use_uncertainty = use_uncertainty

        # Unidirectional GRU cell for causal decoding
        self.gru_cell = nn.GRUCell(d_hidden, d_hidden)

        # Lightweight GCN for spatial coherence per decoded step
        self.spatial_refine = SpatialConvOrderK(
            c_in=d_hidden, c_out=d_hidden,
            support_len=support_len, order=1, include_self=True
        )

        # Output projections
        self.fc_out = nn.Linear(d_hidden, d_out)
        if use_uncertainty:
            self.fc_logvar = nn.Linear(d_hidden, d_out)

        # Gate for spatial residual
        self.gate = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden),
            nn.Sigmoid()
        )

    def forward(self, encoder_hidden, supp):
        """
        Args:
            encoder_hidden: (B, D, N, S) full encoder output
            supp: list of support matrices for GCN [adj_fwd, adj_bwd]
        Returns:
            forecast: (B, H, N, d_out)
            logvar: (B, H, N, d_out) or None
        """
        B, D, N, S = encoder_hidden.shape

        # Initialize from last timestep of encoder
        h = encoder_hidden[:, :, :, -1]  # (B, D, N)
        h = h.permute(0, 2, 1).reshape(B * N, D)  # (B*N, D)

        inp = h.clone()

        forecasts = []
        logvars = []

        for t in range(self.forecast_horizon):
            # GRU step
            h = self.gru_cell(inp, h)  # (B*N, D)

            # Spatial refinement via GCN
            h_spatial = h.reshape(B, N, D).permute(0, 2, 1).unsqueeze(-1)  # (B, D, N, 1)
            h_refined = self.spatial_refine(h_spatial, supp)  # (B, D, N, 1)
            h_refined = h_refined.squeeze(-1).permute(0, 2, 1).reshape(B * N, D)  # (B*N, D)

            # Gated residual: blend GRU hidden with GCN-refined
            gate = self.gate(torch.cat([h, h_refined], dim=-1))
            h = gate * h_refined + (1 - gate) * h

            # Output projection
            h_out = h.reshape(B, N, D)
            out = self.fc_out(h_out)  # (B, N, d_out)
            forecasts.append(out)

            if self.use_uncertainty:
                lv = self.fc_logvar(h_out)
                logvars.append(lv)

            # Next step input = current hidden
            inp = h

        forecast = torch.stack(forecasts, dim=1)  # (B, H, N, d_out)
        logvar = torch.stack(logvars, dim=1) if logvars else None

        return forecast, logvar
