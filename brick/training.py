"""The fixed-budget BRICK objective and sampling schedule."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from .data import prepare
from .model import SpatialConvOrderK


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_plan(data, scene_index, seed):
    rng = np.random.default_rng(seed + data.fold * 7919 + scene_index * 104729)
    times, queries = [], []
    for _ in range(2):
        times.append(rng.integers(data.window - 1, data.train_end - 12, (24, 128)))
        queries.append(rng.choice(data.ki, (24, 128)))
    return np.concatenate(times), np.concatenate(queries)


@contextmanager
def native_only(model):
    before = getattr(model, '_disable_calibration', False)
    model._disable_calibration = True
    try:
        yield
    finally:
        model._disable_calibration = before


def parent_terms(model, data, times, queries):
    times = np.asarray(times)
    queries = np.asarray(queries)
    prepare(data, times, queries, training=True)
    indices = times[:, None] + np.arange(-data.window + 1, 1)[None]
    observed = data.tensor(data.z[indices][:, :, data.ki])[..., None]
    count = len(data.ki)
    ratio = count / data.n + .2 * np.random.random()
    virtual = max(1, int(count / ratio) - count)
    zeros = observed.new_zeros(len(times), data.window, virtual, 1)
    x = torch.cat((observed, zeros), 2)
    mask = torch.cat((torch.ones_like(observed), zeros), 2)
    output = model(x, mask, known_set=data.ki.tolist(), sub_entry_num=virtual, reset=True)
    future = times[:, None] + np.arange(1, 13)[None]
    forecast_target = data.tensor(data.z[future][:, :, data.ki])
    pack = prepare(data, times, queries, training=True)
    model._known_set = None
    supports = SpatialConvOrderK.compute_support(model.adj, data.device)
    prediction, _ = model.impute(pack['physical'], pack['physical_mask'], supports)
    pred = prediction[torch.arange(len(times), device=data.device), :, pack['query'], 0]
    target = data.tensor(data.z[indices, queries[:, None]])
    return dict(reconstruction=(output[0][:, :, :count] - observed).abs().mean(),
                cycle=(output[1] - output[2]).abs().mean(),
                forecast=(output[3][:, :, :count, 0] - forecast_target).abs().mean(),
                history=(pred - target).abs().mean(),
                endpoint=(pred[:, -1] - target[:, -1]).abs().mean())


def calibration_loss(model, data, times, queries):
    pack = prepare(data, times, queries, training=True)
    regions = torch.as_tensor(data.sensor_regions, device=data.device)
    hidden = regions[None] == regions[pack['query'], None]
    mask = pack['physical_mask'].masked_fill(hidden[:, None, :, None], 0)
    keep = mask[:, 0, :, 0].sum(1) >= 2
    if not keep.any():
        return sum(parameter.sum() * 0 for parameter in model.unsupported_calibration_head.parameters())
    cold = dict(physical=pack['physical'][keep].clone(), physical_mask=mask[keep], query=pack['query'][keep].clone())
    ts = np.asarray(times)[keep.cpu().numpy()]
    qs = np.asarray(queries)[keep.cpu().numpy()]
    indices = ts[:, None] + np.arange(-data.window + 1, 1)[None]
    target = data.tensor(data.z[indices, qs[:, None]])
    states = {module: module.training for module in model.modules()}
    known = getattr(model, '_known_set', None)
    flag = getattr(model, '_disable_calibration', False)
    devices = [data.device.index or 0] if data.device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            model._disable_calibration = False
            model._known_set = None
            prediction = model.auxiliary_history(cold)
            return (prediction - target).abs().mean() + 3 * (prediction[:, -1] - target[:, -1]).abs().mean()
    finally:
        for module, training in states.items():
            module.training = training
        model._known_set = known
        model._disable_calibration = flag


def optimizer(model):
    native = [p for name, p in model.named_parameters() if not name.startswith('unsupported_calibration_head.') and name != 'wall_correction']
    return torch.optim.Adam([dict(params=native, lr=.001), dict(params=[model.wall_correction], lr=0.), dict(params=model.unsupported_calibration_head.parameters(), lr=.001)])


def objective(model, data, times, queries):
    with native_only(model):
        terms = parent_terms(model, data, times, queries)
    weights = dict(reconstruction=1., cycle=.25, forecast=.04, history=1., endpoint=3.)
    return sum(weights[key] * value for key, value in terms.items()) + calibration_loss(model, data, times, queries)


def clip_gradients(model):
    native = [p for name, p in model.named_parameters() if not name.startswith('unsupported_calibration_head.')]
    torch.nn.utils.clip_grad_norm_(native, 5., error_if_nonfinite=True)
    torch.nn.utils.clip_grad_norm_(model.unsupported_calibration_head.parameters(), 5., error_if_nonfinite=True)
