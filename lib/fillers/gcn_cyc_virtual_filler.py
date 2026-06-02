import sys
import torch
import numpy as np

from . import Filler
from einops import rearrange
from ..nn.layers.physics_loss import PhysicsLoss, SpatialSmoothLoss, ConnSpatialLoss, ConnGradientLoss


class GCNCycVirtualFiller(Filler):
    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 use_physics_loss=False,
                 physics_loss_weight=0.1,
                 spatial_smooth_weight=0.01,
                 use_uncertainty=False,
                 forecast_loss_weight=0.5,
                 forecast_horizon=0,
                 freeze_encoder=False,
                 use_conn_loss=False,
                 conn_spatial_weight=0.01,
                 conn_gradient_weight=0.01,
                 room_dropout_prob=0.0,
                 room_dropout_ratio=1.0):
        super(GCNCycVirtualFiller, self).__init__(model_class=model_class,
                                                  model_kwargs=model_kwargs,
                                                  optim_class=optim_class,
                                                  optim_kwargs=optim_kwargs,
                                                  loss_fn=loss_fn,
                                                  scaled_target=scaled_target,
                                                  whiten_prob=whiten_prob,
                                                  metrics=metrics,
                                                  scheduler_class=scheduler_class,
                                                  scheduler_kwargs=scheduler_kwargs)

        self.tradeoff = pred_loss_weight
        self.trimming = (warm_up, warm_up)

        self.known_set = None

        # Physics-informed loss modules (Phase 3)
        self.use_physics_loss = use_physics_loss
        self.physics_loss_weight = physics_loss_weight
        self.spatial_smooth_weight = spatial_smooth_weight

        if use_physics_loss:
            self.physics_loss_fn = PhysicsLoss(alpha=0.01, normalize=True)
            self.spatial_smooth_fn = SpatialSmoothLoss()
            print(f"[PI-KITS] Physics loss enabled: "
                  f"weight={physics_loss_weight}, smooth={spatial_smooth_weight}")

        # Conn-based losses (A1: spatial consistency, A2: gradient consistency)
        self.use_conn_loss = use_conn_loss
        self.conn_spatial_weight = conn_spatial_weight
        self.conn_gradient_weight = conn_gradient_weight
        if use_conn_loss:
            self.conn_spatial_fn = ConnSpatialLoss()
            self.conn_gradient_fn = ConnGradientLoss()
            print(f"[PI-KITS] Conn loss enabled: "
                  f"spatial={conn_spatial_weight}, gradient={conn_gradient_weight}")

        # Phase 8: Uncertainty-Aware loss
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            print("[PI-KITS] Uncertainty loss enabled: Gaussian NLL (logvar clamped to [-4, 2])")

        # Phase 11: Forecasting
        self.forecast_loss_weight = forecast_loss_weight
        self.forecast_horizon = forecast_horizon
        self.freeze_encoder = freeze_encoder

        # 两阶段训练: Stage2冻结编码器，只训练forecast decoder
        if self.freeze_encoder:
            frozen_count = 0
            trainable_count = 0
            for name, param in self.model.named_parameters():
                if 'forecast_decoder' in name:
                    param.requires_grad = True
                    trainable_count += 1
                else:
                    param.requires_grad = False
                    frozen_count += 1
            print(f"[PI-KITS] Encoder frozen: {frozen_count} params frozen, "
                  f"{trainable_count} forecast_decoder params trainable")

        # C1: Room dropout augmentation
        self.room_dropout_prob = room_dropout_prob
        self.room_dropout_ratio = room_dropout_ratio
        if room_dropout_prob > 0:
            # Intel Lab room groups (sensor indices)
            self._room_groups = {
                0: [0,1,2,3,4,5,6,7,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41],
                2: [9,10,11,12,13,14,15,16,17,18,19,20,21],
            }
            ratio_str = f"{room_dropout_ratio*100:.0f}%" if room_dropout_ratio < 1.0 else "all"
            print(f"[PI-KITS] Room dropout enabled: prob={room_dropout_prob}, "
                  f"ratio={ratio_str}, rooms=[0({len(self._room_groups[0])}), 2({len(self._room_groups[2])})]")

    def _compute_physics_loss(self, imputation, mask, n_obs):
        """
        计算物理约束损失 (仅在原始观测节点上, 排除虚拟节点)

        Args:
            imputation: (B, S, N_aug, 1) 模型预测 (含虚拟节点)
            mask: (B, S, N_aug, 1) 观测掩码
            n_obs: 原始观测节点数

        Returns:
            physics_loss: 标量
        """
        # 仅取原始观测节点 (排除虚拟节点)
        imp_obs = imputation[:, :, :n_obs, :]
        mask_obs = mask[:, :, :n_obs, :]

        # 获取原始邻接矩阵 (不含虚拟节点增广)
        adj_orig = self.model.adj[:n_obs, :n_obs]

        # 热扩散方程约束
        loss_physics = self.physics_loss_fn(imp_obs, adj_orig, mask_obs)

        # 空间平滑约束
        loss_smooth = torch.tensor(0.0, device=imputation.device)
        if self.spatial_smooth_weight > 0 and hasattr(self.model, 'adj_room'):
            adj_room = self.model.adj_room[:n_obs, :n_obs]
            loss_smooth = self.spatial_smooth_fn(imp_obs, adj_room)

        total = (self.physics_loss_weight * loss_physics +
                 self.spatial_smooth_weight * loss_smooth)
        return total

    def _compute_conn_loss(self, imputation, n_obs):
        """
        计算 conn 加权损失 (A1 + A2)

        A1: conn 加权空间一致性 — conn高的pair温度应接近
        A2: conn 加权梯度一致性 — conn高的pair温度变化趋势应一致

        Args:
            imputation: (B, S, N_aug, C) 模型预测
            n_obs: 原始观测节点数 (排除虚拟节点)
        """
        imp_obs = imputation[:, :, :n_obs, :]
        conn = self.model.conn_matrix[:n_obs, :n_obs]
        dist = (self.model.dist_matrix[:n_obs, :n_obs]
                if hasattr(self.model, 'dist_matrix') else None)

        loss_spatial = self.conn_spatial_fn(imp_obs, conn, dist)
        loss_gradient = self.conn_gradient_fn(imp_obs, conn)

        total = (self.conn_spatial_weight * loss_spatial +
                 self.conn_gradient_weight * loss_gradient)
        return total

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # To make the model inductive
        # => remove unobserved entries from input data and adjacency matrix
        if self.known_set is None:
            # Get observed entries (nonzero masks across time)
            mask = batch_data["mask"]
            # Reduce to node-level for known_set (any channel observed → observed)
            if mask.size(-1) > 1:
                mask_flat = mask.any(dim=-1).float()  # b s n
            else:
                mask_flat = mask.squeeze(-1).float()  # b s n
            mask_flat = rearrange(mask_flat, "b s n -> (b s) n")
            mask_sum = mask_flat.sum(0)  # n
            known_set = torch.where(mask_sum > 0)[0].detach().cpu().numpy().tolist()
            ratio = float(len(known_set) / mask_sum.shape[0])
            self.ratio = ratio
        else:
            known_set = self.known_set

        batch_data["known_set"] = known_set

        x = batch_data["x"]
        mask = batch_data["mask"]
        y = batch_data.pop("y")
        _ = batch_data.pop("eval_mask")  # drop this, we will re-create a new eval_mask (=mask during training)

        # Pop forecast-related keys before model forward (they are not model inputs)
        y_forecast = batch_data.pop('y_forecast', None)
        _ = batch_data.pop('forecast_mask', None)

        x = x[:, :, known_set, :]  # b s n1 d, n1 = num of observed entries
        mask = mask[:, :, known_set, :]  # b s n1 d
        y = y[:, :, known_set, :]  # b s n1 d

        b, s, n, d = mask.size()
        n_obs = n  # 记录原始观测节点数 (用于物理损失计算)

        # C1: Room dropout — hide a random room's data, force cross-wall interpolation
        loss_mask = mask  # default: loss computed on observed positions only
        if self.room_dropout_prob > 0 and self.training and np.random.random() < self.room_dropout_prob:
            room_id = np.random.choice([0, 2])  # skip Room 1 (only 1 sensor)
            room_sensors = self._room_groups[room_id]
            # Find which positions in known_set belong to this room
            drop_indices = [i for i, ks in enumerate(known_set) if ks in room_sensors]
            # Partial room dropout: only drop a fraction of the room's sensors
            if self.room_dropout_ratio < 1.0 and len(drop_indices) > 1:
                n_drop = max(1, int(len(drop_indices) * self.room_dropout_ratio))
                drop_indices = list(np.random.choice(drop_indices, n_drop, replace=False))
            if len(drop_indices) > 0 and len(drop_indices) < n:  # don't drop all
                loss_mask = mask.clone()  # keep original for loss
                mask = mask.clone()
                mask[:, :, drop_indices, :] = 0
                x = x.clone()
                x[:, :, drop_indices, :] = 0

        dynamic_ratio = self.ratio + 0.2 * np.random.random()  # ratio + 0.1
        cur_entry_num = n  # n1
        aug_entry_num = int(cur_entry_num / dynamic_ratio)
        sub_entry_num = aug_entry_num - cur_entry_num  # n2 - n1
        # Dense-sensor batches can leave no room for dynamic virtual expansion.
        # Keep the virtual branch active with one extra entry instead of aborting.
        sub_entry_num = max(1, sub_entry_num)
        self.sub_entry_num = sub_entry_num
        batch_data["reset"] = True

        sub_entry = torch.zeros(b, s, sub_entry_num, d).to(x.device)
        x = torch.cat([x, sub_entry], dim=2)  # b s n2 d
        mask = torch.cat([mask, sub_entry], dim=2).byte()  # b s n2 d
        loss_mask = torch.cat([loss_mask, sub_entry], dim=2).byte()  # b s n2 d
        y = torch.cat([y, sub_entry], dim=2)  # b s n2 d

        eval_mask = loss_mask  # eval on all originally observed positions

        batch_data["x"] = x  # b s n2 d
        batch_data["mask"] = mask  # b s n' 1
        batch_data["sub_entry_num"] = sub_entry_num  # number

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        if self.use_uncertainty:
            imputation, imputation_cyc, target_cyc, logvar = res[0], res[1], res[2], res[3]
            forecast = res[4] if len(res) > 4 else None
            _ = res[5] if len(res) > 5 else None  # forecast_logvar (unused, forecast uses L1)
        else:
            imputation, imputation_cyc, target_cyc = res[0], res[1], res[2]
            logvar = None
            forecast = res[3] if len(res) > 3 else None

        # Extract forecast target from batch (already popped above)
        # y_forecast was popped before predict_batch call

        # trim to imputation horizon len
        imputation, mask, loss_mask, eval_mask, y = self.trim_seq(
            imputation, mask, loss_mask, eval_mask, y)
        imputation_cyc, target_cyc = self.trim_seq(imputation_cyc, target_cyc)
        if logvar is not None:
            logvar = self.trim_seq(logvar)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            imputation_cyc = self._postprocess(imputation_cyc, batch_preprocessing)

        # partial loss + cycle loss
        # Use loss_mask (includes dropped room sensors) for loss computation
        if self.use_uncertainty and logvar is not None:
            # Gaussian NLL with clamped logvar to prevent collapse
            logvar = logvar.clamp(-4, 2)  # prevent NLL degenerate solution
            var = logvar.exp()
            nll = 0.5 * (logvar + (imputation - target) ** 2 / var)
            loss = (nll * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            # Cycle loss stays standard (no uncertainty for reconstruction)
            loss = loss + self.loss_fn(imputation_cyc, target_cyc,
                                       torch.ones_like(imputation_cyc).bool())
            self.log('mean_logvar', logvar.mean().detach(), on_step=False,
                     on_epoch=True, logger=True, prog_bar=False)
        else:
            loss = self.loss_fn(imputation, target, loss_mask) + \
                   1 * self.loss_fn(imputation_cyc, target_cyc,
                                    torch.ones_like(imputation_cyc).bool())

        # Physics-informed loss (Phase 3)
        if self.use_physics_loss:
            loss_physics = self._compute_physics_loss(imputation, mask, n_obs)
            loss = loss + loss_physics
            self.log('physics_loss', loss_physics.detach(), on_step=False,
                     on_epoch=True, logger=True, prog_bar=False)

        # Conn-based losses (A1 spatial + A2 gradient)
        if self.use_conn_loss and hasattr(self.model, 'conn_matrix'):
            loss_conn = self._compute_conn_loss(imputation, n_obs)
            loss = loss + loss_conn
            self.log('conn_loss', loss_conn.detach(), on_step=False,
                     on_epoch=True, logger=True, prog_bar=False)

        # Phase 11: Forecast loss (always L1 — NLL causes degenerate collapse)
        if forecast is not None and y_forecast is not None:
            # Slice forecast target to known_set nodes, pad with zeros for virtual
            y_fc = y_forecast[:, :, known_set, :]
            y_fc = torch.cat([y_fc,
                torch.zeros(b, forecast.size(1), sub_entry_num, d).to(x.device)], dim=2)
            # Mask: only original nodes valid
            fc_mask = torch.zeros_like(y_fc)
            fc_mask[:, :, :n_obs, :] = 1.0

            if self.scaled_target:
                y_fc_target = self._preprocess(y_fc, batch_preprocessing)
            else:
                y_fc_target = y_fc
                forecast = self._postprocess(forecast, batch_preprocessing)

            # Always use L1 for forecast (Gaussian NLL collapses to negative loss)
            loss_forecast = self.loss_fn(forecast, y_fc_target, fc_mask.bool())

            loss = loss + self.forecast_loss_weight * loss_forecast
            self.log('forecast_loss', loss_forecast.detach(), on_step=False,
                     on_epoch=True, logger=True, prog_bar=False)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        # self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data.get('mask')
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        _ = batch_data.pop('y_forecast', None)  # discard forecast target for val
        _ = batch_data.pop('forecast_mask', None)

        # Compute predictions and compute loss
        result = self.predict_batch(batch, preprocess=False, postprocess=False)
        if isinstance(result, (tuple, list)):
            imputation = result[0]
        else:
            imputation = result

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)

        val_loss = self.loss_fn(imputation, target, eval_mask)

        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.val_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        eval_mask = batch_data.pop('eval_mask', None)
        y = batch_data.pop('y')
        y_forecast = batch_data.pop('y_forecast', None)
        _ = batch_data.pop('forecast_mask', None)

        # Direct forward (avoid predict_batch double-unpack reference issues)
        result = self.forward(**batch_data)

        # Unpack: model may return (imputation, logvar, forecast, forecast_logvar) or subsets
        forecast = None
        if isinstance(result, (tuple, list)):
            imputation = result[0]
            idx = 1
            if self.use_uncertainty and len(result) > idx:
                idx += 1  # skip logvar
            if self.forecast_horizon > 0 and len(result) > idx:
                forecast = result[idx]
        else:
            imputation = result

        # Rescale imputation
        imputation = self._postprocess(imputation, batch_preprocessing)

        test_loss = self.loss_fn(imputation, y, eval_mask)

        # Logging
        self.test_metrics.update(imputation.detach(), y, eval_mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_loss', test_loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        # Store forecast for later evaluation (in train.py)
        if forecast is not None:
            forecast = self._postprocess(forecast, batch_preprocessing)
            self._test_forecasts = getattr(self, '_test_forecasts', [])
            self._test_forecasts.append(forecast.detach().cpu())
            if y_forecast is not None:
                # y_forecast is already in original space (like y, never scaled by dataset)
                self._test_forecast_targets = getattr(self, '_test_forecast_targets', [])
                self._test_forecast_targets.append(y_forecast.detach().cpu())

        return test_loss
