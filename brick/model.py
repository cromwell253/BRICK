from einops import rearrange
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ._native.nn.models.kits import KITS
from ._native.nn.layers import SpatialConvOrderK
from ._native.nn.layers.multi_relation_adj import build_multi_relation_adj

def replicate_impute(self, x, mask, supp):
    (b, s, n, c) = x.size()
    imputation = self.relu(self.fc_1(x))
    if self.use_room_embed and hasattr(self, '_room_ids_cur'):
        room_emb = self.room_embed(self._room_ids_cur[:n])
        imputation = imputation + room_emb.unsqueeze(0).unsqueeze(0)
    imputation = rearrange(imputation, 'b s n d -> b d n s')
    d = imputation.size(1)
    t1 = self.t_dims[0]
    imputation = rearrange(imputation, 'b d n s -> b d s n')
    imputation = F.unfold(F.pad(imputation, (0, 0, t1 // 2, t1 // 2), mode='replicate'), kernel_size=(t1, n), padding=(0, 0), stride=(1, 1))
    imputation = imputation.reshape(b, t1 * d, -1, s)
    supp_drop = self.adj_drop(supp, mask)
    imputation = self.relu(self.gcn_1(imputation, supp_drop))
    t2 = self.t_dims[1]
    imputation = rearrange(imputation, 'b d n s -> b d s n')
    imputation = F.unfold(F.pad(imputation, (0, 0, t2 // 2, t2 // 2), mode='replicate'), kernel_size=(t2, n), padding=(0, 0), stride=(1, 1))
    imputation = imputation.reshape(b, t2 * d, -1, s)
    supp_drop = self.adj_drop(supp, mask)
    imputation = self.relu(self.gcn_2(imputation, supp_drop))
    feat = imputation.clone()
    feat = rearrange(feat, 'b d n s -> (b s) n d')
    if mask.size(-1) > 1:
        feat_mask_node = mask.any(dim=-1, keepdim=True).float()
    else:
        feat_mask_node = mask.float()
    feat_mask = rearrange(feat_mask_node, 'b s n d -> (b s) n d')
    if self.use_wall_attn:
        walls_for_attn = None
        if self.walls is not None:
            n_orig = self.walls.size(0)
            if n > n_orig:
                walls_aug = torch.zeros(n, n, device=feat.device)
                walls_aug[:n_orig, :n_orig] = self.walls
                walls_for_attn = walls_aug
            else:
                walls_for_attn = self.walls[:n, :n]
        feat_transfer = self.wall_attention(feat, feat_mask, walls_for_attn)
        feat_transfer = rearrange(feat_transfer, '(b s) n d -> b d n s', b=b, s=s)
    else:
        cosine_eps = 1e-07
        q = feat.clone()
        k = feat.clone().transpose(-2, -1)
        q_norm = torch.norm(q, 2, 2, True)
        k_norm = torch.norm(k, 2, 1, True)
        cos_sim = torch.bmm(q, k) / (torch.bmm(q_norm, k_norm) + cosine_eps)
        cos_sim = (cos_sim + 1.0) / 2.0
        if self.use_conn_donor and hasattr(self, 'conn_matrix'):
            cm = self.conn_matrix
            n_orig = cm.size(0)
            ks = getattr(self, '_known_set', None)
            if self.training and ks is not None:
                cm_slice = cm[ks, :][:, ks]
                n_obs = cm_slice.size(0)
                if n > n_obs:
                    cm_aug = torch.zeros(n, n, device=feat.device)
                    cm_aug[:n_obs, :n_obs] = cm_slice
                else:
                    cm_aug = cm_slice
            elif n > n_orig:
                cm_aug = torch.zeros(n, n, device=feat.device)
                cm_aug[:n_orig, :n_orig] = cm
            else:
                cm_aug = cm[:n, :n]
            if self.training and ks is not None:
                n_obs_actual = len(ks)
            else:
                n_obs_actual = n_orig
            obs_ratio = n_obs_actual / max(n_orig, 1)
            adaptive_scale = max(0.0, min(1.0, (obs_ratio - 0.15) / (0.5 - 0.15)))
            cos_sim = cos_sim + self.conn_donor_bias * adaptive_scale * cm_aug.unsqueeze(0)
        cos_sim_max = cos_sim * feat_mask
        (cos_sim_max_score, cos_sim_max_index) = torch.max(cos_sim_max, dim=1)
        cos_sim_min = cos_sim * (1.0 - feat_mask)
        (cos_sim_min_score, cos_sim_min_index) = torch.max(cos_sim_min, dim=1)
        v = feat.clone().transpose(-2, -1)
        v_unobs = self.bis(v, 2, cos_sim_max_index)
        v_obs = self.bis(v, 2, cos_sim_min_index)
        v_unobs = v_unobs * cos_sim_max_score.unsqueeze(1)
        v_obs = v_obs * cos_sim_min_score.unsqueeze(1)
        if self.use_wall_transfer and hasattr(self, 'wall_count_matrix'):
            wc = self.wall_count_matrix
            n_orig = wc.size(0)
            ks = getattr(self, '_known_set', None)
            if self.training and ks is not None:
                wc_slice = wc[ks, :][:, ks]
                n_obs = wc_slice.size(0)
                if n > n_obs:
                    wc_aug = torch.zeros(n, n, device=feat.device)
                    wc_aug[:n_obs, :n_obs] = wc_slice
                else:
                    wc_aug = wc_slice
            elif n > n_orig:
                wc_aug = torch.zeros(n, n, device=feat.device)
                wc_aug[:n_orig, :n_orig] = wc
            else:
                wc_aug = wc[:n, :n]
            bs_n = cos_sim_max_index.size(0)
            node_idx = torch.arange(n, device=feat.device).unsqueeze(0).expand(bs_n, -1)
            wc_unobs = wc_aug[node_idx, cos_sim_max_index]
            wc_obs = wc_aug[node_idx, cos_sim_min_index]
            corr = self.wall_correction
            v_unobs = v_unobs + corr.unsqueeze(0).unsqueeze(-1) * wc_unobs.unsqueeze(1)
            v_obs = v_obs + corr.unsqueeze(0).unsqueeze(-1) * wc_obs.unsqueeze(1)
        v_unobs = rearrange(v_unobs, '(b s) d n -> b d n s', b=b, s=s)
        v_obs = rearrange(v_obs, '(b s) d n -> b d n s', b=b, s=s)
        feat_mask = rearrange(feat_mask, '(b s) n d -> b d n s', b=b, s=s)
        feat_transfer = v_unobs * (1.0 - feat_mask) + v_obs * feat_mask
    if self.use_wall_attn:
        imputation = feat_transfer
    else:
        imputation = torch.cat([imputation, feat_transfer], dim=1)
        imputation = rearrange(imputation, 'b d n s -> b s n d')
        imputation = self.relu(self.smooth(imputation))
        imputation = rearrange(imputation, 'b s n d -> b d n s')
    if self.use_temporal:
        imputation = self.temporal_gru(imputation)
    t3 = self.t_dims[2]
    imputation = rearrange(imputation, 'b d n s -> b d s n')
    imputation = F.unfold(F.pad(imputation, (0, 0, t3 // 2, t3 // 2), mode='replicate'), kernel_size=(t3, n), padding=(0, 0), stride=(1, 1))
    imputation = imputation.reshape(b, t3 * d, -1, s)
    supp_drop = self.adj_drop(supp, mask)
    imputation = self.relu(self.gcn_3(imputation, supp_drop))
    encoder_hidden = imputation
    imputation = rearrange(imputation, 'b d n s -> b s n d')
    mu = self.fc_2(imputation)
    if self.use_uncertainty:
        logvar = self.fc_logvar(imputation)
        return (mu, logvar, encoder_hidden)
    return (mu, encoder_hidden)

class CommonMode(KITS):

    def __init__(self, data, variant='Common-ScaledResidual'):
        config = json.loads(Path(__file__).with_name('model_config.json').read_text(encoding='utf-8'))
        config.update(_walls_matrix=data.walls, _adj_original=data.adj)
        (adj, _) = build_multi_relation_adj(data.distance, data.conn, np.zeros_like(data.adj), data.walls, thr=0.1, wall_lambda=0.0, room_boost_factor=0.0, add_same_room_edges=False, conn_lambda=0.5)
        super().__init__(adj, 1, 64, SimpleNamespace(**config))
        self.common_variant = variant
        if variant != 'Raw-Replay':
            nn.init.zeros_(self.fc_2.weight)
            nn.init.zeros_(self.fc_2.bias)

    def decompose(self, x, mask):
        clean = torch.where(mask.bool(), x, torch.zeros_like(x))
        count = mask.to(x.dtype).sum(2, keepdim=True)
        assert (count > 0).all(), 'At least one visible support per time required'
        level = clean.sum(2, keepdim=True) / count
        if self.common_variant == 'Common-ScaledResidual':
            variance = torch.where(mask.bool(), (clean - level).square(), torch.zeros_like(clean)).sum(2, keepdim=True) / count
            scale = (variance + 1e-06).sqrt().clamp_min(0.1)
        else:
            scale = torch.ones_like(level)
        residual = torch.where(mask.bool(), (clean - level) / scale, torch.zeros_like(clean))
        return (residual, level, scale)

    def impute(self, x, mask, supp):
        if self.common_variant == 'Raw-Replay':
            return super().impute(x, mask, supp)
        (residual, level, scale) = self.decompose(x, mask)
        (prediction, hidden) = super().impute(residual, mask, supp)
        return (prediction * scale + level, hidden)

class Boundary(CommonMode):

    def impute(self, x, mask, supp):
        (residual, level, scale) = self.decompose(x, mask)
        (prediction, hidden) = replicate_impute(self, residual, mask, supp)
        return (prediction * scale + level, hidden)

class LocalAnchor(Boundary):
    anchor_mix = 0.5

    def decompose_with_support(self, x, mask, supp):
        (residual, global_level, scale) = super().decompose(x, mask)
        if self.anchor_mix == 0.0:
            return (residual, global_level.expand_as(x), scale)
        assert self.anchor_mix == 0.5
        clean = torch.where(mask.bool(), x, torch.zeros_like(x))
        weights = supp[0] + supp[1]
        assert weights.shape == (x.shape[2], x.shape[2]) and torch.isfinite(weights).all() and (weights >= 0).all()
        weights = weights - torch.diag_embed(weights.diagonal())
        count = torch.einsum('ij,btjc->btic', weights, mask.to(x.dtype))
        numerator = torch.einsum('ij,btjc->btic', weights, clean)
        local = torch.where(count > 1e-08, numerator / count.clamp_min(1e-08), global_level)
        level = 0.5 * global_level + 0.5 * local
        residual = torch.where(mask.bool(), (clean - level) / scale, torch.zeros_like(x))
        return (residual, level, scale)

    def impute(self, x, mask, supp):
        (residual, level, scale) = self.decompose_with_support(x, mask, supp)
        (prediction, hidden) = replicate_impute(self, residual, mask, supp)
        return (prediction * scale + level, hidden)

class ComponentResidual(LocalAnchor):
    component_center = True

    def __init__(self, data, variant):
        super().__init__(data, variant)
        (multiplier, penalty, nugget) = data.args.kriging_parameters
        self._physical_n = data.n
        self._nugget = float(nugget)
        self._kernel = np.exp(-data.distance.astype(np.float64) / (data.length * multiplier))
        self._kernel *= np.exp(-penalty * (data.sensor_regions[:, None] != data.sensor_regions[None, :]))
        self._tables = {}

    def table(self, available):
        key = tuple(np.flatnonzero(available).tolist())
        if key in self._tables:
            return self._tables[key]
        ids = np.asarray(key, dtype=int)
        assert len(ids) > 0
        result = np.zeros((self._physical_n, self._physical_n), np.float64)
        if len(ids) == 1:
            result[:, ids] = 1.0
        else:
            count = len(ids)
            lhs = np.zeros((count + 1, count + 1))
            lhs[:-1, :-1] = self._kernel[np.ix_(ids, ids)] + np.eye(count) * self._nugget
            lhs[-1, :-1] = 1.0
            lhs[:-1, -1] = 1.0
            rhs = np.vstack((self._kernel[:, ids].T, np.ones((1, self._physical_n))))
            result[:, ids] = np.linalg.solve(lhs, rhs)[:-1].T
            inverse = np.linalg.inv(lhs)
            for (i, query) in enumerate(ids):
                result[query, ids] = -inverse[i, :count] / inverse[i, i]
                result[query, query] = 0.0
        assert np.isfinite(result).all() and np.allclose(result.sum(1), 1.0, atol=2e-05)
        result = result.astype(np.float32)
        self._tables[key] = result
        return result

    def decompose_with_support(self, x, mask, supp):
        if not self.component_center or getattr(self, '_known_set', None) is not None or x.shape[2] != self._physical_n:
            return super().decompose_with_support(x, mask, supp)
        available = mask[:, 0, :, 0].bool()
        assert torch.equal(mask.bool(), available[:, None, :, None].expand_as(mask))
        (_, _, scale) = self.decompose(x, mask)
        clean = torch.where(mask.bool(), x, torch.zeros_like(x))
        matrices = np.stack([self.table(row) for row in available.detach().cpu().numpy()])
        weights = torch.as_tensor(matrices, device=x.device, dtype=x.dtype)
        level = torch.einsum('bij,btjc->btic', weights, clean)
        residual = torch.where(mask.bool(), (clean - level) / scale, torch.zeros_like(x))
        return (residual, level, scale)

class ResidualScale(ComponentResidual):
    residual_scaling = True

    def decompose_with_support(self, x, mask, supp):
        (residual, level, scale) = super().decompose_with_support(x, mask, supp)
        if not self.residual_scaling or getattr(self, '_known_set', None) is not None or x.shape[2] != self._physical_n:
            return (residual, level, scale)
        clean = torch.where(mask.bool(), x, torch.zeros_like(x))
        count = mask.to(x.dtype).sum(2, keepdim=True)
        rms = (torch.where(mask.bool(), (clean - level).square(), torch.zeros_like(x)).sum(2, keepdim=True) / count + 1e-06).sqrt()
        scale = rms.clamp_min(0.1)
        return (torch.where(mask.bool(), (clean - level) / scale, torch.zeros_like(x)), level, scale)

class QueryReadout(ResidualScale):

    def __init__(self, data, variant):
        super().__init__(data, variant)
        with torch.random.fork_rng(devices=[]):
            self.query_local_head = nn.Linear(64, 1)
            nn.init.zeros_(self.query_local_head.weight)
            nn.init.zeros_(self.query_local_head.bias)

    def impute(self, x, mask, supp):
        (residual, level, scale) = self.decompose_with_support(x, mask, supp)
        local = []

        def capture(module, args, result):
            local.append(self.relu(result.clone()))
        hook = self.smooth.register_forward_hook(capture)
        try:
            (prediction, hidden) = replicate_impute(self, residual, mask, supp)
        finally:
            hook.remove()
        assert len(local) == 1
        correction = self.query_local_head(local[0])
        correction = torch.where(mask.bool(), torch.zeros_like(correction), correction)
        return ((prediction + correction) * scale + level, hidden)

class SupportCalibration(QueryReadout):

    def __init__(self, data, variant):
        super().__init__(data, variant)
        self._calibration_regions = np.asarray(data.sensor_regions).copy()
        with torch.random.fork_rng(devices=[]):
            self.unsupported_calibration_head = nn.Linear(64, 1)
            nn.init.zeros_(self.unsupported_calibration_head.weight)
            nn.init.zeros_(self.unsupported_calibration_head.bias)

    def _calibration_forward(self, x, mask, supp):
        physical = getattr(self, '_known_set', None) is None and x.shape[2] == self._physical_n
        captured = []

        def capture(module, args, result):
            captured.append(self.relu(result.detach().clone()))
        hook = self.smooth.register_forward_hook(capture) if physical else None
        try:
            (reference, hidden) = super().impute(x, mask, supp)
        finally:
            if hook is not None:
                hook.remove()
        if not physical:
            return (reference, hidden, torch.zeros_like(reference))
        assert len(captured) == 1
        available = mask[:, 0, :, 0].bool()
        assert torch.equal(mask.bool(), available[:, None, :, None].expand_as(mask))
        regions = torch.as_tensor(self._calibration_regions, device=x.device)
        same = (regions[:, None] == regions[None, :]).to(x.dtype)
        count = torch.einsum('bn,qn->bq', available.to(x.dtype), same)
        gate = (count == 0) & ~available
        with torch.no_grad():
            (_, _, scale) = self.decompose_with_support(x, mask, supp)
        raw = self.unsupported_calibration_head(captured[0])
        branch = torch.where(gate[:, None, :, None], raw, torch.zeros_like(raw)) * scale.detach()
        return (reference, hidden, branch)

    def impute(self, x, mask, supp):
        (reference, hidden, branch) = self._calibration_forward(x, mask, supp)
        return (reference + branch, hidden)

    def auxiliary_history(self, pack):
        assert not self.training, 'Calibration supervision uses evaluation-mode features'
        self._known_set = None
        x = torch.where(pack['physical_mask'].bool(), pack['physical'], torch.zeros_like(pack['physical']))
        supp = SpatialConvOrderK.compute_support(self.adj, x.device)
        (reference, _, branch) = self._calibration_forward(x, pack['physical_mask'], supp)
        pred = reference.detach() + branch
        return pred[torch.arange(len(x), device=x.device), :, pack['query'], 0]

class BRICK(SupportCalibration):

    def impute(self, x, mask, supp):
        if getattr(self, '_disable_calibration', False):
            return QueryReadout.impute(self, x, mask, supp)
        return super().impute(x, mask, supp)

def build(data):
    return BRICK(data, 'Common-ScaledResidual').to(data.device)
