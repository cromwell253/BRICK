import sys
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from ..layers import SpatialConvOrderK
from ..layers.temporal_module import TemporalGRU
from ..layers.wall_attention import WallAwareAttention
from ..layers.physics_conv import PhysicsGuidedConv

class KITS(nn.Module):

    def __init__(self, adj, d_in, d_hidden, args):
        super(KITS, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dataset_name = args.dataset_name
        self.t_dim = 3
        self.register_buffer('adj', torch.tensor(adj).float())
        self.fc_1 = nn.Linear(d_in, d_hidden)
        use_multiscale = getattr(args, 'use_multiscale_temporal', False)
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.t_dims = [3, 5, 7]
            pass
        else:
            self.t_dims = [self.t_dim, self.t_dim, self.t_dim]
        self.gcn_1 = SpatialConvOrderK(c_in=self.t_dims[0] * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_2 = SpatialConvOrderK(c_in=self.t_dims[1] * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_3 = SpatialConvOrderK(c_in=self.t_dims[2] * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        use_phys_conv = getattr(args, 'use_physics_conv', False)
        self.use_phys_conv = use_phys_conv
        if use_phys_conv:
            self.gcn_1 = PhysicsGuidedConv(c_in=self.t_dims[0] * d_hidden, c_out=d_hidden, support_len=2, order=1, include_self=False)
            pass
        self.smooth = nn.Linear(2 * d_hidden, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, d_in)
        use_uncertainty = getattr(args, 'use_uncertainty', False)
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.fc_logvar = nn.Linear(d_hidden, d_in)
            pass
        use_wall_attn = getattr(args, 'use_wall_attention', False)
        self.use_wall_attn = use_wall_attn
        if use_wall_attn:
            n_heads = getattr(args, 'wall_attn_heads', 4)
            attn_drop = getattr(args, 'wall_attn_dropout', 0.1)
            self.wall_attention = WallAwareAttention(d_hidden=d_hidden, n_heads=n_heads, dropout=attn_drop)
            walls_np = getattr(args, '_walls_matrix', None)
            if walls_np is not None:
                self.register_buffer('walls', torch.tensor(walls_np, dtype=torch.float32))
            else:
                self.walls = None
            pass
        else:
            self.walls = None
        use_wall_transfer = getattr(args, 'use_wall_transfer', False)
        self.use_wall_transfer = use_wall_transfer
        if use_wall_transfer and (not use_wall_attn):
            walls_np = getattr(args, '_walls_matrix', None)
            if walls_np is not None:
                max_walls = int(walls_np.max())
                self.register_buffer('wall_count_matrix', torch.tensor(walls_np, dtype=torch.float32))
                self.wall_correction = nn.Parameter(torch.zeros(1))
                pass
            else:
                self.use_wall_transfer = False
                pass
        use_conn_donor = getattr(args, 'use_conn_donor', False)
        self.use_conn_donor = use_conn_donor
        if use_conn_donor:
            conn_np = getattr(args, '_conn_matrix', None)
            if conn_np is not None:
                conn_norm = conn_np / (conn_np.max() + 1e-08)
                self.register_buffer('conn_matrix', torch.tensor(conn_norm, dtype=torch.float32))
                self.conn_donor_bias = nn.Parameter(torch.tensor(0.1))
                pass
            else:
                self.use_conn_donor = False
                pass
        use_conn_loss = getattr(args, 'use_conn_loss', False)
        if use_conn_loss:
            conn_np = getattr(args, '_conn_matrix', None)
            dist_np = getattr(args, '_dist_matrix', None)
            if conn_np is not None:
                conn_norm = conn_np / (conn_np.max() + 1e-08)
                if not hasattr(self, 'conn_matrix'):
                    self.register_buffer('conn_matrix', torch.tensor(conn_norm, dtype=torch.float32))
                if dist_np is not None:
                    self.register_buffer('dist_matrix', torch.tensor(dist_np, dtype=torch.float32))
                pass
            else:
                pass
        use_temporal = getattr(args, 'use_temporal_gru', False)
        self.use_temporal = use_temporal
        if use_temporal:
            gru_layers = getattr(args, 'gru_num_layers', 1)
            self.temporal_gru = TemporalGRU(d_hidden=d_hidden, num_layers=gru_layers, bidirectional=True)
            pass
        self.relu = nn.ReLU(inplace=True)
        self.supp = None
        self.adj_aug = None
        self.obs_neighbors = None
        if args.use_adj_drop:
            pass
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = nn.Identity()
        if args.use_init:
            pass
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)
        self.learnable_conn = getattr(args, 'learnable_conn_lambda', False)
        self.use_room_embed = getattr(args, 'use_room_embed', False)
        if self.use_room_embed:
            room_ids_np = getattr(args, '_room_ids', None)
            if room_ids_np is not None:
                num_rooms = int(room_ids_np.max()) + 1
                self.register_buffer('room_ids', torch.tensor(room_ids_np, dtype=torch.long))
                self.room_embed = nn.Embedding(num_rooms + 1, d_hidden)
                nn.init.normal_(self.room_embed.weight, std=0.02)
                self._virtual_room_id = num_rooms
                pass
            else:
                self.use_room_embed = False
                pass
        if self.learnable_conn:
            A_dist_raw = getattr(args, '_adj_dist_raw', None)
            A_conn_norm = getattr(args, '_adj_conn_norm', None)
            A_room_raw = getattr(args, '_adj_room_raw', None)
            if A_dist_raw is not None and A_conn_norm is not None:
                self.register_buffer('A_dist_raw', torch.tensor(A_dist_raw, dtype=torch.float32))
                self.register_buffer('A_conn_norm', torch.tensor(A_conn_norm, dtype=torch.float32))
                if A_room_raw is not None:
                    self.register_buffer('A_room_raw', torch.tensor(A_room_raw, dtype=torch.float32))
                else:
                    self.A_room_raw = None
                init_lambda = getattr(args, 'conn_lambda', 0.5)
                init_rb = getattr(args, 'room_boost_factor', 0.3)
                self._conn_lambda_logit = nn.Parameter(torch.tensor(np.log(init_lambda / (1 - init_lambda + 1e-08))))
                self._room_boost_raw = nn.Parameter(torch.tensor(np.log(np.exp(init_rb) - 1 + 1e-08)))
                self._adj_thr = getattr(args, 'adj_threshold', 0.1)
                pass
            else:
                self.learnable_conn = False
                pass
        self.forecast_horizon = getattr(args, 'forecast_horizon', 0)
        self.forecast_detach = getattr(args, 'forecast_detach', True)
        self.forecast_use_orig_adj = getattr(args, 'forecast_use_orig_adj', False)
        self.forecast_residual = getattr(args, 'forecast_residual', False)
        self.forecast_residual_init = getattr(args, 'forecast_residual_init', 0.1)
        self.forecast_residual_detach_context = getattr(args, 'forecast_residual_detach_context', False)
        if self.forecast_horizon > 0:
            from ..layers.forecast_decoder import ForecastDecoder
            self.forecast_decoder = ForecastDecoder(d_hidden=d_hidden, d_out=d_in, forecast_horizon=self.forecast_horizon, support_len=2, use_uncertainty=self.use_uncertainty)
            if self.forecast_use_orig_adj:
                adj_orig = getattr(args, '_adj_original', None)
                if adj_orig is not None:
                    self.register_buffer('adj_orig', torch.tensor(adj_orig, dtype=torch.float32))
                    pass
                else:
                    self.forecast_use_orig_adj = False
                    pass
            detach_str = 'detach' if self.forecast_detach else 'end-to-end'
            adj_str = 'orig_adj' if self.forecast_use_orig_adj else 'same_adj'
            pass
            if self.forecast_residual:
                self.forecast_residual_scale = nn.Parameter(torch.tensor(float(self.forecast_residual_init)))
                pass

    def adj_drop(self, supp, mask):
        mask_reduced = mask.sum(dim=-1)
        mask_reduced = rearrange(mask_reduced, 'b s n -> (b s) n')
        mask_reduced = mask_reduced.sum(0)
        obs_index = mask_reduced > 0
        supp_update = []
        for i in range(len(supp)):
            s = supp[i].clone().detach()
            s_hor = s[obs_index, :]
            s_ver = s[:, obs_index]
            s_hor = self.dropout(s_hor)
            s_ver = self.dropout(s_ver)
            s[obs_index, :] = s_hor
            s[:, obs_index] = s_ver
            supp_update.append(s)
        return supp_update

    def _build_learnable_adj(self):
        pass
        conn_lambda = torch.sigmoid(self._conn_lambda_logit)
        room_boost = F.softplus(self._room_boost_raw)
        conn_factor = 1.0 - conn_lambda + conn_lambda * self.A_conn_norm
        adj = self.A_dist_raw * conn_factor
        if self.A_room_raw is not None:
            adj = adj * (1.0 + room_boost * self.A_room_raw)
        adj = adj * (adj >= self._adj_thr).float()
        adj = adj * (1.0 - torch.eye(adj.size(0), device=adj.device))
        return adj

    def forward(self, x, mask=None, known_set=None, sub_entry_num=None, reset=False):
        if self.learnable_conn:
            adj = self._build_learnable_adj()
        else:
            adj = self.adj.clone()
        if self.training:
            if reset:
                adj = adj[known_set, :]
                adj = adj[:, known_set]
                n1 = adj.size(0)
                if self.obs_neighbors is None:
                    obs_neighbors = {}
                    for i in range(n1):
                        row_nonzero = set(torch.where(adj[i, :] > 0)[0].detach().cpu().numpy().tolist())
                        col_nonzero = set(torch.where(adj[:, i] > 0)[0].detach().cpu().numpy().tolist())
                        nonzero = row_nonzero.union(col_nonzero)
                        obs_neighbors[i] = list(nonzero)
                    self.obs_neighbors = obs_neighbors
                else:
                    obs_neighbors = self.obs_neighbors
                n2 = n1 + sub_entry_num
                adj_aug = torch.rand(n2, n2).to(adj.device)
                adj_aug[:n1, :n1] = adj
                adj_diag = 1.0 - torch.eye(n2).to(adj.device)
                adj_aug = adj_aug * adj_diag
                neighbors = copy.deepcopy(obs_neighbors)
                adj_aug_mask = torch.zeros_like(adj_aug)
                adj_aug_mask[:n1, :n1] = 1
                for i in range(n1, n2):
                    n_current = range(len(neighbors.keys()))
                    rand_entry = random.sample(n_current, 1)[0]
                    rand_neighbors = neighbors[rand_entry]
                    p = np.random.rand(1)
                    valid_neighbors = (np.random.rand(len(rand_neighbors)) < p).astype(int)
                    valid_neighbors = np.where(valid_neighbors == 1)[0].tolist()
                    valid_neighbors = [rand_neighbors[idx] for idx in valid_neighbors]
                    all_entries = [rand_entry]
                    all_entries.extend(valid_neighbors)
                    for entry in all_entries:
                        neighbors[entry].append(i)
                    neighbors[i] = all_entries
                    options = [0, 1, 2]
                    connected_conditions = [random.choice(options) for _ in range(len(all_entries))]
                    for j in range(len(all_entries)):
                        entry = all_entries[j]
                        condition = connected_conditions[j]
                        if condition == 0 or condition == 2:
                            adj_aug_mask[entry, i] = 1
                        if condition == 1 or condition == 2:
                            adj_aug_mask[i, entry] = 1
                adj_aug = adj_aug * adj_aug_mask
                if self.dataset_name in ['sea_loop_point']:
                    adj_aug[adj_aug > 0] = 1
                self.adj_aug = adj_aug
            else:
                adj_aug = self.adj_aug
            adj = adj_aug.detach()
        if self.learnable_conn and self.training and (known_set is not None):
            adj_learn = self._build_learnable_adj()
            adj_learn_sub = adj_learn[known_set, :][:, known_set]
            n1 = len(known_set)
            n2 = adj.size(0)
            if n2 > n1:
                adj_padded = adj.clone()
                adj_padded[:n1, :n1] = adj_learn_sub
                adj = adj_padded
            else:
                adj = adj_learn_sub
        supp = SpatialConvOrderK.compute_support(adj, x.device)
        self._known_set = known_set
        if self.use_room_embed and known_set is not None:
            room_ids_cur = self.room_ids[known_set]
            if sub_entry_num is not None and sub_entry_num > 0:
                virtual_ids = torch.full((sub_entry_num,), self._virtual_room_id, dtype=torch.long, device=room_ids_cur.device)
                room_ids_cur = torch.cat([room_ids_cur, virtual_ids])
            self._room_ids_cur = room_ids_cur
        elif self.use_room_embed:
            self._room_ids_cur = self.room_ids
        result = self.impute(x, mask, supp)
        if self.use_uncertainty:
            (imputation, logvar, encoder_hidden) = result
        else:
            (imputation, encoder_hidden) = result
            logvar = None
        forecast = None
        forecast_logvar = None
        if self.forecast_horizon > 0 and hasattr(self, 'forecast_decoder'):
            fc_hidden = encoder_hidden.detach() if self.forecast_detach else encoder_hidden
            if self.forecast_use_orig_adj and hasattr(self, 'adj_orig'):
                if self.training and known_set is not None:
                    ks = known_set
                    adj_o = self.adj_orig[ks, :][:, ks]
                    n1 = len(ks)
                    n2 = n1 + (sub_entry_num or 0)
                    if n2 > n1:
                        adj_padded = torch.zeros(n2, n2, device=x.device)
                        adj_padded[:n1, :n1] = adj_o
                        adj_o = adj_padded
                else:
                    adj_o = self.adj_orig
                fc_supp = SpatialConvOrderK.compute_support(adj_o, x.device)
            else:
                fc_supp = supp
            (forecast, forecast_logvar) = self.forecast_decoder(fc_hidden, fc_supp)
            if self.forecast_residual:
                last_context = torch.where(mask.bool(), x, imputation)[:, -1:, :, :]
                if self.forecast_residual_detach_context:
                    last_context = last_context.detach()
                forecast = last_context + self.forecast_residual_scale * forecast
        if not self.training:
            imputation = torch.where(mask.bool(), x, imputation)
            if forecast is not None:
                if logvar is not None:
                    return (imputation, logvar, forecast, forecast_logvar)
                return (imputation, forecast, forecast_logvar)
            if logvar is not None:
                return (imputation, logvar)
            return imputation
        else:
            y = torch.where(mask.bool(), x, imputation)
            x = imputation * (1 - mask)
            result_cyc = self.impute(x, 1 - mask, supp)
            if self.use_uncertainty:
                (imputation_cyc, logvar_cyc, _) = result_cyc
                if forecast is not None:
                    return (imputation, imputation_cyc, y, logvar, forecast, forecast_logvar)
                return (imputation, imputation_cyc, y, logvar)
            else:
                (imputation_cyc, _) = result_cyc
                if forecast is not None:
                    return (imputation, imputation_cyc, y, forecast, forecast_logvar)
                return (imputation, imputation_cyc, y)

    def encode_for_field(self, x, mask):
        pass
        if x.dim() != 4 or mask.shape != x.shape:
            raise ValueError('encode_for_field expects x and mask with shape (B,S,N,C)')
        adj = self._build_learnable_adj() if self.learnable_conn else self.adj
        supp = SpatialConvOrderK.compute_support(adj, x.device)
        self._known_set = None
        result = self.impute(x, mask, supp)
        if self.use_uncertainty:
            (imputation, _, encoder_hidden) = result
        else:
            (imputation, encoder_hidden) = result
        reconstructed_nodes = torch.where(mask.bool(), x, imputation)
        return (reconstructed_nodes, encoder_hidden)

    def impute(self, x, mask, supp):
        (b, s, n, c) = x.size()
        imputation = self.relu(self.fc_1(x))
        if self.use_room_embed and hasattr(self, '_room_ids_cur'):
            room_emb = self.room_embed(self._room_ids_cur[:n])
            imputation = imputation + room_emb.unsqueeze(0).unsqueeze(0)
        imputation = rearrange(imputation, 'b s n d -> b d n s')
        d = imputation.size(1)
        t1 = self.t_dims[0]
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(t1, n), padding=(t1 // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, t1 * d, -1, s)
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_1(imputation, supp_drop))
        t2 = self.t_dims[1]
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(t2, n), padding=(t2 // 2, 0), stride=(1, 1))
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
        imputation = F.unfold(imputation, kernel_size=(t3, n), padding=(t3 // 2, 0), stride=(1, 1))
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

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument('--d-hidden', type=int, default=64)
        return parser
