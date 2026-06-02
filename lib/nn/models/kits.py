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
    def __init__(self,
                 adj,
                 d_in,
                 d_hidden,
                 args
                 ):
        super(KITS, self).__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.dataset_name = args.dataset_name

        self.t_dim = 3
        self.register_buffer('adj', torch.tensor(adj).float())
        self.fc_1 = nn.Linear(d_in, d_hidden)

        # Phase 7: Multi-Scale Temporal Pyramid
        use_multiscale = getattr(args, 'use_multiscale_temporal', False)
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.t_dims = [3, 5, 7]  # local, medium, long-range
            print(f"[PI-KITS] Multi-scale temporal: kernels={self.t_dims}")
        else:
            self.t_dims = [self.t_dim, self.t_dim, self.t_dim]

        self.gcn_1 = SpatialConvOrderK(c_in=self.t_dims[0] * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_2 = SpatialConvOrderK(c_in=self.t_dims[1] * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)
        self.gcn_3 = SpatialConvOrderK(c_in=self.t_dims[2] * d_hidden, c_out=d_hidden, support_len=2 * 1, order=1, include_self=False)

        # Phase 6: Physics-Guided Conv (replaces gcn_1)
        use_phys_conv = getattr(args, 'use_physics_conv', False)
        self.use_phys_conv = use_phys_conv
        if use_phys_conv:
            self.gcn_1 = PhysicsGuidedConv(
                c_in=self.t_dims[0] * d_hidden, c_out=d_hidden,
                support_len=2, order=1, include_self=False)
            print(f"[PI-KITS] Physics-Guided Conv enabled: replaces gcn_1")

        self.smooth = nn.Linear(2 * d_hidden, d_hidden)
        self.fc_2 = nn.Linear(d_hidden, d_in)

        # Phase 8: Uncertainty-Aware output
        use_uncertainty = getattr(args, 'use_uncertainty', False)
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.fc_logvar = nn.Linear(d_hidden, d_in)
            print("[PI-KITS] Uncertainty output enabled: mu + log_var")

        # Phase 5: Wall-Aware Cross-Attention (replaces Hard Transfer)
        use_wall_attn = getattr(args, 'use_wall_attention', False)
        self.use_wall_attn = use_wall_attn
        if use_wall_attn:
            n_heads = getattr(args, 'wall_attn_heads', 4)
            attn_drop = getattr(args, 'wall_attn_dropout', 0.1)
            self.wall_attention = WallAwareAttention(
                d_hidden=d_hidden, n_heads=n_heads, dropout=attn_drop)
            # Load walls matrix if available
            walls_np = getattr(args, '_walls_matrix', None)
            if walls_np is not None:
                self.register_buffer(
                    'walls', torch.tensor(walls_np, dtype=torch.float32))
            else:
                self.walls = None
            print(f"[PI-KITS] Wall-Aware Attention enabled: "
                  f"heads={n_heads}, dropout={attn_drop}")
        else:
            self.walls = None

        # Wall-Corrected Transfer: learn temperature correction per wall crossing
        # Instead of penalizing cross-wall donors, correct the transferred value
        # Physics: walls cause temperature OFFSET (insulation), not signal loss
        use_wall_transfer = getattr(args, 'use_wall_transfer', False)
        self.use_wall_transfer = use_wall_transfer
        if use_wall_transfer and not use_wall_attn:
            walls_np = getattr(args, '_walls_matrix', None)
            if walls_np is not None:
                max_walls = int(walls_np.max())
                # Register wall count matrix as buffer
                self.register_buffer(
                    'wall_count_matrix',
                    torch.tensor(walls_np, dtype=torch.float32))
                # Learnable correction: scalar offset per wall crossing
                # Physics: each wall shifts ALL hidden features by the same amount
                # Only 1 param → robust, no overfitting risk
                self.wall_correction = nn.Parameter(torch.zeros(1))
                print(f"[PI-KITS] Wall-Corrected Transfer enabled: "
                      f"max_walls={max_walls}, 1 learnable param")
            else:
                self.use_wall_transfer = False
                print("[PI-KITS] Wall-Corrected Transfer: no walls matrix, disabled")

        # Conn-Guided Donor Selection: bias cos_sim toward high-connectivity donors
        # Physics: nodes with higher connection probability (same room, fewer walls)
        # are better temperature references
        use_conn_donor = getattr(args, 'use_conn_donor', False)
        self.use_conn_donor = use_conn_donor
        if use_conn_donor:
            conn_np = getattr(args, '_conn_matrix', None)
            if conn_np is not None:
                conn_norm = conn_np / (conn_np.max() + 1e-8)
                self.register_buffer(
                    'conn_matrix',
                    torch.tensor(conn_norm, dtype=torch.float32))
                # Learnable bias: how much connectivity influences donor selection
                self.conn_donor_bias = nn.Parameter(torch.tensor(0.1))
                print(f"[PI-KITS] Conn-Guided Donor Selection enabled: 1 learnable param")
            else:
                self.use_conn_donor = False
                print("[PI-KITS] Conn-Guided Donor: no conn matrix, disabled")

        # Conn/Dist buffers for conn-based losses (A1 spatial + A2 gradient)
        # These are stored unconditionally when available, used by filler
        use_conn_loss = getattr(args, 'use_conn_loss', False)
        if use_conn_loss:
            conn_np = getattr(args, '_conn_matrix', None)
            dist_np = getattr(args, '_dist_matrix', None)
            if conn_np is not None:
                conn_norm = conn_np / (conn_np.max() + 1e-8)
                if not hasattr(self, 'conn_matrix'):
                    self.register_buffer(
                        'conn_matrix',
                        torch.tensor(conn_norm, dtype=torch.float32))
                if dist_np is not None:
                    self.register_buffer(
                        'dist_matrix',
                        torch.tensor(dist_np, dtype=torch.float32))
                print(f"[PI-KITS] Conn loss buffers registered (conn + dist)")
            else:
                print("[PI-KITS] Warning: conn loss enabled but conn.npy not found")

        # Phase 4: Temporal GRU module (after cross-reference, before gcn_3)
        use_temporal = getattr(args, 'use_temporal_gru', False)
        self.use_temporal = use_temporal
        if use_temporal:
            gru_layers = getattr(args, 'gru_num_layers', 1)
            self.temporal_gru = TemporalGRU(
                d_hidden=d_hidden,
                num_layers=gru_layers,
                bidirectional=True
            )
            print(f"[PI-KITS] Temporal GRU enabled: layers={gru_layers}, bidirectional")

        self.relu = nn.ReLU(inplace=True)
        self.supp = None
        self.adj_aug = None
        self.obs_neighbors = None

        if args.use_adj_drop:
            print("use adj dropout...")
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = nn.Identity()

        if args.use_init:
            print("use init...")
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight, gain=1)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        # Phase 12: Learnable conn_lambda — dynamically blend A_dist and A_conn
        self.learnable_conn = getattr(args, 'learnable_conn_lambda', False)

        # D3: Room-aware positional encoding
        self.use_room_embed = getattr(args, 'use_room_embed', False)
        if self.use_room_embed:
            room_ids_np = getattr(args, '_room_ids', None)
            if room_ids_np is not None:
                num_rooms = int(room_ids_np.max()) + 1
                self.register_buffer('room_ids', torch.tensor(room_ids_np, dtype=torch.long))
                # +1 for virtual nodes (room_id = num_rooms)
                self.room_embed = nn.Embedding(num_rooms + 1, d_hidden)
                nn.init.normal_(self.room_embed.weight, std=0.02)
                self._virtual_room_id = num_rooms
                print(f"[PI-KITS] Room embedding enabled: {num_rooms} rooms + 1 virtual, dim={d_hidden}")
            else:
                self.use_room_embed = False
                print("[PI-KITS] Room embedding: no room_ids, disabled")
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
                # Learnable parameters: conn_lambda ∈ (0,1) via sigmoid, room_boost ≥ 0 via softplus
                self._conn_lambda_logit = nn.Parameter(torch.tensor(
                    np.log(init_lambda / (1 - init_lambda + 1e-8))))  # inverse sigmoid
                self._room_boost_raw = nn.Parameter(torch.tensor(
                    np.log(np.exp(init_rb) - 1 + 1e-8)))  # inverse softplus
                self._adj_thr = getattr(args, 'adj_threshold', 0.1)
                print(f"[PI-KITS] Learnable conn_lambda enabled: "
                      f"init_lambda={init_lambda:.2f}, init_room_boost={init_rb:.2f}")
            else:
                self.learnable_conn = False
                print("[PI-KITS] Learnable conn: missing raw matrices, disabled")

        # Phase 11: Temporal Forecasting
        self.forecast_horizon = getattr(args, 'forecast_horizon', 0)
        self.forecast_detach = getattr(args, 'forecast_detach', True)
        self.forecast_use_orig_adj = getattr(args, 'forecast_use_orig_adj', False)
        self.forecast_residual = getattr(args, 'forecast_residual', False)
        self.forecast_residual_init = getattr(args, 'forecast_residual_init', 0.1)
        self.forecast_residual_detach_context = getattr(
            args, 'forecast_residual_detach_context', False)
        if self.forecast_horizon > 0:
            from ..layers.forecast_decoder import ForecastDecoder
            self.forecast_decoder = ForecastDecoder(
                d_hidden=d_hidden,
                d_out=d_in,
                forecast_horizon=self.forecast_horizon,
                support_len=2,
                use_uncertainty=self.use_uncertainty
            )
            # 方案B: 存储原始距离邻接供forecast专用
            if self.forecast_use_orig_adj:
                adj_orig = getattr(args, '_adj_original', None)
                if adj_orig is not None:
                    self.register_buffer('adj_orig',
                        torch.tensor(adj_orig, dtype=torch.float32))
                    print(f"[PI-KITS] Forecast uses ORIGINAL distance adjacency (separate from interpolation)")
                else:
                    self.forecast_use_orig_adj = False
                    print(f"[PI-KITS] Warning: _adj_original not provided, forecast uses same adj")
            detach_str = "detach" if self.forecast_detach else "end-to-end"
            adj_str = "orig_adj" if self.forecast_use_orig_adj else "same_adj"
            print(f"[PI-KITS] Forecasting enabled: horizon={self.forecast_horizon}, {detach_str}, {adj_str}")
            if self.forecast_residual:
                self.forecast_residual_scale = nn.Parameter(
                    torch.tensor(float(self.forecast_residual_init)))
                print(f"[PI-KITS] Residual forecasting enabled: "
                      f"forecast = last_context + scale * delta, "
                      f"init_scale={self.forecast_residual_init:.3f}, "
                      f"detach_context={self.forecast_residual_detach_context}")

    def adj_drop(self, supp, mask):
        # supp: list, fwd and bwd adj - (n, n)
        # mask: b, s, n, d_in (d_in=1 or 2)
        # Reduce across channels: node is observed if any channel is observed
        mask_reduced = mask.sum(dim=-1)  # b, s, n
        mask_reduced = rearrange(mask_reduced, 'b s n -> (b s) n')
        mask_reduced = mask_reduced.sum(0)  # n
        obs_index = mask_reduced > 0  # n
        # unobs_index = mask == 0  # n
        supp_update = []
        for i in range(len(supp)):
            s = supp[i].clone().detach()

            s_hor = s[obs_index, :]  # n1, n
            s_ver = s[:, obs_index]  # n, n1

            s_hor = self.dropout(s_hor)
            s_ver = self.dropout(s_ver)

            s[obs_index, :] = s_hor
            s[:, obs_index] = s_ver

            supp_update.append(s)
        return supp_update

    def _build_learnable_adj(self):
        """Dynamically compute adjacency using learnable conn_lambda and room_boost."""
        conn_lambda = torch.sigmoid(self._conn_lambda_logit)
        room_boost = F.softplus(self._room_boost_raw)

        # A = A_dist * ((1-λ) + λ * A_conn_norm) * (1 + rb * A_room)
        conn_factor = (1.0 - conn_lambda) + conn_lambda * self.A_conn_norm
        adj = self.A_dist_raw * conn_factor
        if self.A_room_raw is not None:
            adj = adj * (1.0 + room_boost * self.A_room_raw)

        # Threshold for sparsity
        adj = adj * (adj >= self._adj_thr).float()
        # Remove self-loops
        adj = adj * (1.0 - torch.eye(adj.size(0), device=adj.device))
        return adj

    def forward(self, x, mask=None, known_set=None, sub_entry_num=None, reset=False):
        # Dynamic adjacency if learnable conn enabled, else static
        if self.learnable_conn:
            adj = self._build_learnable_adj()
        else:
            adj = self.adj.clone()  # adjacency matrix

        if self.training:
            if reset:
                # ========================================
                # Obtain 1-hop neighbors of each observed entry
                # ========================================
                # preserve observed entries
                adj = adj[known_set, :]
                adj = adj[:, known_set]  # n1, n1
                n1 = adj.size(0)

                # get the 1-hop neighbors of each observed entry.
                if self.obs_neighbors is None:
                    obs_neighbors = {}
                    for i in range(n1):
                        row_nonzero = set(torch.where(adj[i, :] > 0)[0].detach().cpu().numpy().tolist())
                        col_nonzero = set(torch.where(adj[:, i] > 0)[0].detach().cpu().numpy().tolist())
                        nonzero = row_nonzero.union(col_nonzero)
                        obs_neighbors[i] = list(nonzero)  # 1-hop neighbors
                    self.obs_neighbors = obs_neighbors
                else:
                    obs_neighbors = self.obs_neighbors  # n1, n1, note that cannot use copy!!!

                # ========================================
                # Create dynamic adjacency matrix
                # ========================================
                # initialize dynamic adjacency matrix
                n2 = n1 + sub_entry_num
                adj_aug = torch.rand(n2, n2).to(adj.device)  # n2, n2

                # preserve original observed parts in newly-created adj
                adj_aug[:n1, :n1] = adj

                # remove self-loop
                adj_diag = 1. - torch.eye(n2).to(adj.device)  # n2, n2
                adj_aug = adj_aug * adj_diag  # n2, n2

                # for each newly-created virtual entry, randomly connect it to one observed entry
                neighbors = copy.deepcopy(obs_neighbors)  # initially has n1 entries' 1-hop neighbors
                adj_aug_mask = torch.zeros_like(adj_aug)  # n2, n2
                adj_aug_mask[:n1, :n1] = 1
                for i in range(n1, n2):
                    n_current = range(len(neighbors.keys()))  # number of current entries (obs and already added virtual)
                    rand_entry = random.sample(n_current, 1)[0]  # randomly sample 1 entry (obs or already added virtual)
                    rand_neighbors = neighbors[rand_entry]  # get 1-hop neighbors of sampled entry
                    p = np.random.rand(1)  # randomly generate a probability

                    # randomly select neighbors
                    valid_neighbors = (np.random.rand(len(rand_neighbors)) < p).astype(int)
                    valid_neighbors = np.where(valid_neighbors == 1)[0].tolist()
                    valid_neighbors = [rand_neighbors[idx] for idx in valid_neighbors]
                    all_entries = [rand_entry]
                    all_entries.extend(valid_neighbors)

                    # add current virtual entry to the 1-hop neighbors of selected entries
                    for entry in all_entries:
                        neighbors[entry].append(i)

                    # add selected entries to the 1-hop neighbors of current virtual entry
                    neighbors[i] = all_entries

                    options = [0, 1, 2]  # 0: forward; 1: backward; 2: bi-direction
                    connected_conditions = [random.choice(options) for _ in range(len(all_entries))]
                    for j in range(len(all_entries)):
                        entry = all_entries[j]
                        condition = connected_conditions[j]

                        if condition == 0 or condition == 2:
                            adj_aug_mask[entry, i] = 1
                        if condition == 1 or condition == 2:
                            adj_aug_mask[i, entry] = 1

                adj_aug = adj_aug * adj_aug_mask

                if self.dataset_name in ["sea_loop_point"]:
                    adj_aug[adj_aug > 0] = 1  # only for sea-loop, because their adj are binary

                self.adj_aug = adj_aug
            else:
                adj_aug = self.adj_aug
            adj = adj_aug.detach()

        # For learnable conn: also compute a differentiable support from learnable adj
        # so that gradients can flow to conn_lambda and room_boost parameters.
        # The virtual-node augmented adj is detached (random parts are non-differentiable),
        # but we add a parallel path through the learnable adj for gradient computation.
        if self.learnable_conn and self.training and known_set is not None:
            # Build differentiable adj for observed nodes only
            adj_learn = self._build_learnable_adj()
            adj_learn_sub = adj_learn[known_set, :][:, known_set]
            n1 = len(known_set)
            n2 = adj.size(0)
            if n2 > n1:
                # Pad to match virtual-node augmented size
                adj_padded = adj.clone()  # detached aug adj
                # Replace observed portion with differentiable version
                adj_padded[:n1, :n1] = adj_learn_sub
                adj = adj_padded
            else:
                adj = adj_learn_sub

        supp = SpatialConvOrderK.compute_support(adj, x.device)

        # Store known_set for wall-aware Hard Transfer in impute()
        self._known_set = known_set

        # D3: Build room IDs for current node set (known_set + virtual nodes)
        if self.use_room_embed and known_set is not None:
            room_ids_cur = self.room_ids[known_set]
            if sub_entry_num is not None and sub_entry_num > 0:
                virtual_ids = torch.full((sub_entry_num,), self._virtual_room_id,
                                        dtype=torch.long, device=room_ids_cur.device)
                room_ids_cur = torch.cat([room_ids_cur, virtual_ids])
            self._room_ids_cur = room_ids_cur
        elif self.use_room_embed:
            self._room_ids_cur = self.room_ids

        result = self.impute(x, mask, supp)
        if self.use_uncertainty:
            imputation, logvar, encoder_hidden = result
        else:
            imputation, encoder_hidden = result
            logvar = None

        # Phase 11: Forecasting
        forecast = None
        forecast_logvar = None
        if self.forecast_horizon > 0 and hasattr(self, 'forecast_decoder'):
            # 方案A: detach控制 — False时forecast loss参与训练编码器
            fc_hidden = encoder_hidden.detach() if self.forecast_detach else encoder_hidden
            # 方案B: 原始邻接 — forecast decoder用未修改的距离邻接
            if self.forecast_use_orig_adj and hasattr(self, 'adj_orig'):
                if self.training and known_set is not None:
                    # 训练时: adj_orig是全N×N, 需切片到known_set再pad虚拟节点
                    ks = known_set
                    adj_o = self.adj_orig[ks, :][:, ks]  # n1 × n1
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
            forecast, forecast_logvar = self.forecast_decoder(fc_hidden, fc_supp)
            if self.forecast_residual:
                # Forecast the future change around the latest reconstructed field.
                last_context = torch.where(mask.bool(), x, imputation)[:, -1:, :, :]
                if self.forecast_residual_detach_context:
                    last_context = last_context.detach()
                forecast = last_context + self.forecast_residual_scale * forecast

        if not self.training:
            imputation = torch.where(mask.bool(), x, imputation)
            if forecast is not None:
                if logvar is not None:
                    return imputation, logvar, forecast, forecast_logvar
                return imputation, forecast, forecast_logvar
            if logvar is not None:
                return imputation, logvar
            return imputation
        else:
            y = torch.where(mask.bool(), x, imputation)
            x = imputation * (1 - mask)
            result_cyc = self.impute(x, 1 - mask, supp)
            if self.use_uncertainty:
                imputation_cyc, logvar_cyc, _ = result_cyc
                if forecast is not None:
                    return imputation, imputation_cyc, y, logvar, forecast, forecast_logvar
                return imputation, imputation_cyc, y, logvar
            else:
                imputation_cyc, _ = result_cyc
                if forecast is not None:
                    return imputation, imputation_cyc, y, forecast, forecast_logvar
                return imputation, imputation_cyc, y

    def impute(self, x, mask, supp):
        b, s, n, c = x.size()
        imputation = self.relu(self.fc_1(x))  # bs, s, n, dim

        # D3: Add room positional encoding
        if self.use_room_embed and hasattr(self, '_room_ids_cur'):
            room_emb = self.room_embed(self._room_ids_cur[:n])  # (n, d_hidden)
            imputation = imputation + room_emb.unsqueeze(0).unsqueeze(0)  # broadcast (1,1,n,d)

        imputation = rearrange(imputation, 'b s n d -> b d n s')
        d = imputation.size(1)

        # Layer 1: t_dims[0] kernel
        t1 = self.t_dims[0]
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(t1, n), padding=(t1 // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, t1 * d, -1, s)
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_1(imputation, supp_drop))

        # Layer 2: t_dims[1] kernel
        t2 = self.t_dims[1]
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(t2, n), padding=(t2 // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, t2 * d, -1, s)
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_2(imputation, supp_drop))

        # ========================================
        # Cross-Reference: Wall-Aware Attention or Hard Transfer
        # ========================================
        # b d n s
        feat = imputation.clone()
        feat = rearrange(feat, 'b d n s -> (b s) n d')
        # Reduce multi-channel mask to node-level (BS, N, 1) for cross-reference
        if mask.size(-1) > 1:
            feat_mask_node = mask.any(dim=-1, keepdim=True).float()  # (b, s, n, 1)
        else:
            feat_mask_node = mask.float()  # (b, s, n, 1)
        feat_mask = rearrange(feat_mask_node, 'b s n d -> (b s) n d')

        if self.use_wall_attn:
            # Phase 5: Wall-Aware Soft Attention
            # Get walls matrix, handling augmented nodes (virtual nodes)
            walls_for_attn = None
            if self.walls is not None:
                n_orig = self.walls.size(0)
                if n > n_orig:
                    # Pad walls matrix for virtual nodes (no wall penalty)
                    walls_aug = torch.zeros(n, n, device=feat.device)
                    walls_aug[:n_orig, :n_orig] = self.walls
                    walls_for_attn = walls_aug
                else:
                    walls_for_attn = self.walls[:n, :n]

            feat_transfer = self.wall_attention(feat, feat_mask, walls_for_attn)
            feat_transfer = rearrange(feat_transfer, '(b s) n d -> b d n s', b=b, s=s)
        else:
            # Original Hard Transfer
            cosine_eps = 1e-7
            q = feat.clone()  # b n d
            k = feat.clone().transpose(-2, -1)  # b d n
            q_norm = torch.norm(q, 2, 2, True)
            k_norm = torch.norm(k, 2, 1, True)

            cos_sim = torch.bmm(q, k) / (torch.bmm(q_norm, k_norm) + cosine_eps)
            cos_sim = (cos_sim + 1.) / 2.

            # Conn-Guided Donor Selection: add connectivity bias to cos_sim
            # High conn (same room) → prefer as donor; Low conn (cross-wall) → less preferred
            if self.use_conn_donor and hasattr(self, 'conn_matrix'):
                cm = self.conn_matrix  # (N_orig, N_orig)
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
                else:
                    if n > n_orig:
                        cm_aug = torch.zeros(n, n, device=feat.device)
                        cm_aug[:n_orig, :n_orig] = cm
                    else:
                        cm_aug = cm[:n, :n]

                # Adaptive scaling: full strength when sensors plentiful,
                # gracefully decay to 0 when sensors are sparse
                # obs_ratio ≥ 0.5 → scale=1.0 (full v8 advantage)
                # obs_ratio = 0.3 → scale≈0.43 (moderate)
                # obs_ratio ≤ 0.15 → scale=0.0 (pure cosine, v7c behavior)
                if self.training and ks is not None:
                    n_obs_actual = len(ks)
                else:
                    n_obs_actual = n_orig
                obs_ratio = n_obs_actual / max(n_orig, 1)
                adaptive_scale = max(0.0, min(1.0,
                    (obs_ratio - 0.15) / (0.5 - 0.15)))

                # Add learnable bias with adaptive scaling
                cos_sim = cos_sim + self.conn_donor_bias * adaptive_scale * cm_aug.unsqueeze(0)

            # Wall-Corrected Transfer: correction is applied AFTER donor selection

            cos_sim_max = cos_sim * feat_mask
            cos_sim_max_score, cos_sim_max_index = torch.max(cos_sim_max, dim=1)
            cos_sim_min = cos_sim * (1. - feat_mask)
            cos_sim_min_score, cos_sim_min_index = torch.max(cos_sim_min, dim=1)

            v = feat.clone().transpose(-2, -1)
            v_unobs = self.bis(v, 2, cos_sim_max_index)
            v_obs = self.bis(v, 2, cos_sim_min_index)
            v_unobs = v_unobs * cos_sim_max_score.unsqueeze(1)
            v_obs = v_obs * cos_sim_min_score.unsqueeze(1)

            # Wall-Corrected Transfer: apply learnable correction based on wall count
            # between each node and its selected donor
            if self.use_wall_transfer and hasattr(self, 'wall_count_matrix'):
                wc = self.wall_count_matrix  # (N_orig, N_orig)
                n_orig = wc.size(0)
                ks = getattr(self, '_known_set', None)

                if self.training and ks is not None:
                    wc_slice = wc[ks, :][:, ks]  # (n_obs, n_obs)
                    n_obs = wc_slice.size(0)
                    if n > n_obs:
                        wc_aug = torch.zeros(n, n, device=feat.device)
                        wc_aug[:n_obs, :n_obs] = wc_slice
                    else:
                        wc_aug = wc_slice
                else:
                    if n > n_orig:
                        wc_aug = torch.zeros(n, n, device=feat.device)
                        wc_aug[:n_orig, :n_orig] = wc
                    else:
                        wc_aug = wc[:n, :n]

                # Get wall count between each node and its donor
                # cos_sim_max_index: (BS, N) — donor index for unobserved nodes
                # cos_sim_min_index: (BS, N) — donor index for observed nodes
                bs_n = cos_sim_max_index.size(0)
                node_idx = torch.arange(n, device=feat.device).unsqueeze(0).expand(bs_n, -1)

                # Wall count to donor for unobserved (obs→unobs transfer)
                wc_unobs = wc_aug[node_idx, cos_sim_max_index]  # (BS, N)
                # Wall count to donor for observed (unobs→obs transfer)
                wc_obs = wc_aug[node_idx, cos_sim_min_index]  # (BS, N)

                # Correction: wall_correction (D,) * wall_count → per-feature offset
                # Physics: each wall crossing shifts hidden features by a learned vector
                corr = self.wall_correction  # (D,)

                # Correction for unobserved nodes: (BS, N) x (D,) → (BS, D, N)
                v_unobs = v_unobs + corr.unsqueeze(0).unsqueeze(-1) * wc_unobs.unsqueeze(1)

                # Correction for observed nodes
                v_obs = v_obs + corr.unsqueeze(0).unsqueeze(-1) * wc_obs.unsqueeze(1)

            v_unobs = rearrange(v_unobs, '(b s) d n -> b d n s', b=b, s=s)
            v_obs = rearrange(v_obs, '(b s) d n -> b d n s', b=b, s=s)

            feat_mask = rearrange(feat_mask, '(b s) n d -> b d n s', b=b, s=s)
            feat_transfer = v_unobs * (1. - feat_mask) + v_obs * feat_mask

        if self.use_wall_attn:
            # Wall-Aware Attention already produces d_hidden output with residual
            imputation = feat_transfer
        else:
            # Original: concat + smooth
            imputation = torch.cat([imputation, feat_transfer], dim=1)
            imputation = rearrange(imputation, 'b d n s -> b s n d')
            imputation = self.relu(self.smooth(imputation))
            imputation = rearrange(imputation, 'b s n d -> b d n s')

        # Phase 4: Temporal GRU enhancement
        if self.use_temporal:
            imputation = self.temporal_gru(imputation)  # (B, D, N, S) -> (B, D, N, S)

        # ========================================
        # Output
        # ========================================
        t3 = self.t_dims[2]
        imputation = rearrange(imputation, 'b d n s -> b d s n')
        imputation = F.unfold(imputation, kernel_size=(t3, n), padding=(t3 // 2, 0), stride=(1, 1))
        imputation = imputation.reshape(b, t3 * d, -1, s)
        supp_drop = self.adj_drop(supp, mask)
        imputation = self.relu(self.gcn_3(imputation, supp_drop))

        # Save encoder hidden for forecast decoder (before output projection)
        encoder_hidden = imputation  # (B, D, N, S)

        imputation = rearrange(imputation, 'b d n s -> b s n d')
        mu = self.fc_2(imputation)  # b s n d_in

        if self.use_uncertainty:
            logvar = self.fc_logvar(imputation)  # b s n d_in
            return mu, logvar, encoder_hidden

        return mu, encoder_hidden

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
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
