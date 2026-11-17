"""Query inference without access to query labels."""
import numpy as np
import torch
from .data import prepare
from .model import SpatialConvOrderK


@torch.no_grad()
def predict(model, data, times, queries, batch_size=16):
    model.eval()
    model._known_set = None
    model._disable_calibration = False
    outputs = []
    supports = SpatialConvOrderK.compute_support(model.adj, data.device)
    for start in range(0, len(times), batch_size):
        pack = prepare(data, times[start:start + batch_size], queries[start:start + batch_size])
        prediction, _ = model.impute(pack['physical'], pack['physical_mask'], supports)
        selected = prediction[torch.arange(len(pack['query']), device=data.device), -1, pack['query'], 0]
        outputs.append(selected.cpu().numpy().astype(np.float64) * data.std + data.mean)
    return np.concatenate(outputs)
