"""Published building tables and fixed cold-query partitions."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import torch


class BuildingData:
    def __init__(self, root, scene, fold=0, device='cpu'):
        self.root = Path(root)
        rows = json.loads((self.root / 'configs/splits.json').read_text(encoding='utf-8'))
        self.split = next(row for row in rows if row['scene'] == scene and row['fold'] == fold)
        self.scene = scene
        self.fold = fold
        self.device = torch.device(device)
        folder = self.root / 'data' / scene
        nodes = pd.read_csv(folder / 'coordinates.csv').sort_values('node_id')
        if nodes.node_id.tolist() != list(range(len(nodes))):
            raise ValueError('Node identifiers must be contiguous and zero based')
        self.coords = nodes[['x_m', 'y_m']].to_numpy(dtype=np.float32)
        self.sensor_regions = nodes.component_id.to_numpy(dtype=np.int64)
        self.n = len(nodes)
        self.distance = np.linalg.norm(self.coords[:, None] - self.coords[None], axis=-1)
        self.length = float(np.median(self.distance[self.distance > 0]))
        pairs = pd.read_csv(folder / 'connection_probability.csv')
        self.conn = np.zeros((self.n, self.n), dtype=np.float32)
        self.walls = np.zeros_like(self.conn)
        self.conn[pairs.source_node, pairs.target_node] = pairs.connection_probability
        self.walls[pairs.source_node, pairs.target_node] = pairs.wall_count
        self.adj = np.exp(-(self.distance / max(float(self.distance.std()), 1e-6))**2).astype(np.float32)
        self.adj[self.adj < .1] = 0
        np.fill_diagonal(self.adj, 0)
        self.args = SimpleNamespace(kriging_parameters=(self.split['kernel_length_multiplier'], self.split['component_penalty'], self.split['nugget']))
        self.ki = np.asarray(self.split['train_nodes'], dtype=int)
        self.val_queries = np.asarray(self.split['development_query_nodes'], dtype=int)
        self.window = 24
        self.train_end = 708
        self.val_end = 808
        self.val_times = np.arange(731, 808)[np.linspace(0, 76, 24, dtype=int)]
        # Development queries are read only by the scoring entry point.
        raw = pd.read_csv(folder / 'temperature.csv', usecols=[str(i) for i in self.ki], nrows=self.val_end)
        observations = raw[[str(i) for i in self.ki]].to_numpy(dtype=np.float32)
        self.t = len(observations)
        normalizers = json.loads((self.root / 'configs/source_normalizers.json').read_text(encoding='utf-8'))
        normalizer = next(row for row in normalizers if row['scene'] == scene and row['fold'] == fold)
        self.mean = normalizer['mean']
        self.std = normalizer['std']
        self.z = np.zeros((self.t, self.n), dtype=np.float32)
        self.z[:, self.ki] = (observations - self.mean) / self.std

    def tensor(self, values):
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)


def prepare(data, times, queries, training=False):
    times = np.asarray(times, dtype=int)
    queries = np.asarray(queries, dtype=int)
    if times.shape != queries.shape or times.ndim != 1:
        raise ValueError('One query is required for each endpoint')
    if times.min() < data.window - 1:
        raise ValueError('Insufficient history')
    if training and (times.max() + 12 >= data.train_end or not np.isin(queries, data.ki).all()):
        raise ValueError('Training queries and targets must lie inside the training partition')
    indices = times[:, None] + np.arange(-data.window + 1, 1)[None]
    physical = data.tensor(data.z[indices])[..., None]
    mask = torch.zeros_like(physical)
    donor_ids = []
    for i, query in enumerate(queries):
        donors = np.setdiff1d(data.ki, [query])
        order = np.argsort(data.distance[query, donors], kind='stable')
        donors = donors[order][:len(data.ki) - 1]
        mask[i, :, donors] = 1
        donor_ids.append(donors.tolist())
    physical = torch.where(mask.bool(), physical, torch.zeros_like(physical))
    return dict(physical=physical, physical_mask=mask, query=torch.as_tensor(queries, device=data.device), donor_ids=donor_ids)
