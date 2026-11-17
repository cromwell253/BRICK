pass
import torch
import torch.nn as nn
from .spatial_conv import SpatialConvOrderK

class ForecastDecoder(nn.Module):

    def __init__(self, d_hidden, d_out, forecast_horizon, support_len=2, use_uncertainty=False):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.forecast_horizon = forecast_horizon
        self.use_uncertainty = use_uncertainty
        self.gru_cell = nn.GRUCell(d_hidden, d_hidden)
        self.spatial_refine = SpatialConvOrderK(c_in=d_hidden, c_out=d_hidden, support_len=support_len, order=1, include_self=True)
        self.fc_out = nn.Linear(d_hidden, d_out)
        if use_uncertainty:
            self.fc_logvar = nn.Linear(d_hidden, d_out)
        self.gate = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.Sigmoid())

    def forward(self, encoder_hidden, supp):
        pass
        (B, D, N, S) = encoder_hidden.shape
        h = encoder_hidden[:, :, :, -1]
        h = h.permute(0, 2, 1).reshape(B * N, D)
        inp = h.clone()
        forecasts = []
        logvars = []
        for t in range(self.forecast_horizon):
            h = self.gru_cell(inp, h)
            h_spatial = h.reshape(B, N, D).permute(0, 2, 1).unsqueeze(-1)
            h_refined = self.spatial_refine(h_spatial, supp)
            h_refined = h_refined.squeeze(-1).permute(0, 2, 1).reshape(B * N, D)
            gate = self.gate(torch.cat([h, h_refined], dim=-1))
            h = gate * h_refined + (1 - gate) * h
            h_out = h.reshape(B, N, D)
            out = self.fc_out(h_out)
            forecasts.append(out)
            if self.use_uncertainty:
                lv = self.fc_logvar(h_out)
                logvars.append(lv)
            inp = h
        forecast = torch.stack(forecasts, dim=1)
        logvar = torch.stack(logvars, dim=1) if logvars else None
        return (forecast, logvar)
