from torchmetrics import Metric
from losses import MQLoss
import torch
import warnings


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary function to handle divide by 0
    """
    # More robust: handle both zeros and very small denominators
    mask = torch.abs(b) < 1e-8
    result = torch.where(mask, torch.zeros_like(a), a / b)
    return result


class Coverage(Metric):
    def __init__(self, dist_sync_on_step=False, level=0.95):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.level_low = (1.0-level)/2
        self.level_high = 1.0 - self.level_low
        self.level = level

    def add_evaluation_quantiles(self, quantiles: torch.Tensor):
        quantiles_metric = torch.Tensor([(1 - (1-self.level)/2), (1-self.level)/2])
        quantiles_metric = torch.repeat_interleave(quantiles_metric[None], repeats=quantiles.shape[0], dim=0)
        quantiles_metric = quantiles_metric.to(quantiles)
        return torch.cat([quantiles, quantiles_metric], dim=-1)

    def update(self, preds: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> None:
        """ Compute coverage metric for prediction intervals

        :param preds: Bx..xQ tensor of predicted values, Q is the number of quantiles
        :param target: Bx..x1, or Bx.. tensor of target values
        :param q: Bx..xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value if multi-quantile loss function
        """
        if target.dim() != preds.dim():
            target = target[..., None]

        # Robustly find the nearest quantile indices for the desired coverage levels
        # q may be shaped BxQ, Bx1xQ or BxHxQ. Normalize to BxQ for index selection
        q_sel = q
        if q_sel.dim() == 3 and q_sel.shape[1] == 1:
            # Bx1xQ -> BxQ
            q_sel = q_sel[:, 0, :]
        elif q_sel.dim() == 3 and q_sel.shape[1] > 1:
            # BxHxQ -> use the first horizon's quantiles (assume same across horizons)
            q_sel = q_sel[:, 0, :]

        # Compute nearest indices for high and low levels
        # q_sel: BxQ
        with torch.no_grad():
            diff_high = torch.abs(q_sel - self.level_high)
            diff_low = torch.abs(q_sel - self.level_low)
            idx_high = torch.argmin(diff_high, dim=-1)  # shape: (B,)
            idx_low = torch.argmin(diff_low, dim=-1)

        # Build gather indices to select predictions at those quantile positions
        # preds shape: B x ... x Q
        B = preds.shape[0]
        spatial_shape = preds.shape[1:-1]  # can be () or (H,)
        # Create gather index shape: [B] + [1]*len(spatial_shape) + [1]
        gather_shape = [B] + [1] * len(spatial_shape) + [1]
        idx_high_g = idx_high.view(*gather_shape).expand(*([B] + list(spatial_shape) + [1])).to(preds.device)
        idx_low_g = idx_low.view(*gather_shape).expand(*([B] + list(spatial_shape) + [1])).to(preds.device)

        preds_high = preds.gather(-1, idx_high_g)
        preds_low = preds.gather(-1, idx_low_g)

        # Now compute coverage: target must match preds spatial dims. Ensure target has same rank
        if target.dim() != preds_high.dim():
            target = target[..., None]

        # Use min/max to handle non-monotonic quantiles (low might be > high)
        preds_upper = torch.maximum(preds_high, preds_low)
        preds_lower = torch.minimum(preds_high, preds_low)
        
        # Filter out NaN/Inf values - only count valid entries
        valid_mask = ~(torch.isnan(preds_upper) | torch.isinf(preds_upper) | 
                       torch.isnan(preds_lower) | torch.isinf(preds_lower) |
                       torch.isnan(target) | torch.isinf(target))
        
        if valid_mask.any():
            # Coverage: target is in interval [lower, upper)
            in_interval = (target >= preds_lower) & (target <= preds_upper)
            self.numerator += (in_interval & valid_mask).sum()
            self.denominator += valid_mask.sum()

    def compute(self):
        if self.denominator > 0:
            return self.numerator / self.denominator
        else:
            return torch.tensor(float('nan'))


class CRPS(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon
        self.mqloss = MQLoss()

    def update(self, preds: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> None:
        """ Compute multi-quantile loss function

        :param preds: BxHxQ tensor of predicted values, Q is the number of quantiles
        :param target: BxHx1, or BxH tensor of target values
        :param q: BxHxQ or Bx1xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value if multi-quantile loss function
        """
        try:
            # Check for NaN/Inf in inputs
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                warnings.warn(f"CRPS: NaN or Inf detected in predictions (nan: {torch.isnan(preds).sum()}, inf: {torch.isinf(preds).sum()})")
                return
            if torch.isnan(target).any() or torch.isinf(target).any():
                warnings.warn(f"CRPS: NaN or Inf detected in targets (nan: {torch.isnan(target).sum()}, inf: {torch.isinf(target).sum()})")
                return
            if torch.isnan(q).any() or torch.isinf(q).any():
                warnings.warn("CRPS: NaN or Inf detected in quantiles")
                return
            
            if self.horizon is None:
                loss_val = self.mqloss(input=preds, target=target, q=q)
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    warnings.warn("CRPS: MQLoss returned NaN or Inf")
                    return
                    
                self.numerator += loss_val * torch.numel(preds)
                self.denominator += torch.numel(preds)
            else:
                # Convert horizon to integer if it's a tensor or ensure it's a valid index
                if isinstance(self.horizon, torch.Tensor):
                    horizon_idx = int(self.horizon.item())
                elif isinstance(self.horizon, (list, tuple)):
                    horizon_idx = self.horizon[0] if len(self.horizon) > 0 else 0
                else:
                    horizon_idx = int(self.horizon)
                
                # Ensure horizon_idx is within bounds
                if horizon_idx < 0 or horizon_idx >= preds.shape[1]:
                    warnings.warn(f"CRPS: Invalid horizon {horizon_idx} for preds shape {preds.shape}, using all horizons")
                    loss_val = self.mqloss(input=preds, target=target, q=q)
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        return
                    self.numerator += loss_val * torch.numel(preds)
                    self.denominator += torch.numel(preds)
                else:
                    # Handle q tensor indexing - it might be Bx1xQ or BxHxQ
                    if q.shape[1] == 1:
                        # q is Bx1xQ, use it as is for all horizons
                        q_indexed = q[:, 0, :]
                    else:
                        # q is BxHxQ, index the specific horizon
                        q_indexed = q[:, horizon_idx, :]
                    
                    loss_val = self.mqloss(
                        input=preds[:, horizon_idx, :], 
                        target=target[:, horizon_idx],
                        q=q_indexed
                    )
                    
                    if torch.isnan(loss_val) or torch.isinf(loss_val):
                        warnings.warn(f"CRPS: MQLoss returned NaN or Inf at horizon {horizon_idx}")
                        return
                    
                    self.numerator += loss_val * torch.numel(preds[:, horizon_idx, :])
                    self.denominator += torch.numel(preds[:, horizon_idx, :])
                    
        except Exception as e:
            # More informative error logging
            warnings.warn(f"CRPS update failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    def compute(self):
        if self.denominator > 0:
            result = 2 * (self.numerator / self.denominator)
            if torch.isnan(result) or torch.isinf(result):
                warnings.warn("CRPS: Final computation resulted in NaN or Inf")
                return torch.tensor(float('nan'))
            return result
        else:
            return torch.tensor(float('nan'))


class MAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("mape_sum", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        
        # Check for NaN/Inf
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            warnings.warn("MAPE: NaN or Inf in predictions, skipping update")
            return
        if torch.isnan(target).any() or torch.isinf(target).any():
            warnings.warn("MAPE: NaN or Inf in targets, skipping update")
            return
        
        if self.horizon is None:
            mape = _divide_no_nan(torch.abs(target - preds), torch.abs(target))
            # Filter out invalid values
            valid_mask = ~(torch.isnan(mape) | torch.isinf(mape))
            if valid_mask.any():
                self.mape_sum += mape[valid_mask].sum()
                self.nsamples += valid_mask.sum()
        else:
            mape = _divide_no_nan(torch.abs(target[:, self.horizon] - preds[:, self.horizon]),
                                   torch.abs(target[:, self.horizon]))
            valid_mask = ~(torch.isnan(mape) | torch.isinf(mape))
            if valid_mask.any():
                self.mape_sum += mape[valid_mask].sum()
                self.nsamples += valid_mask.sum()

    def compute(self):
        if self.nsamples > 0:
            return 100 * (self.mape_sum / self.nsamples)
        else:
            return torch.tensor(float('nan'))


class SMAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("smape", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        
        # Check for NaN/Inf
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            warnings.warn("SMAPE: NaN or Inf in predictions, skipping update")
            return
        if torch.isnan(target).any() or torch.isinf(target).any():
            warnings.warn("SMAPE: NaN or Inf in targets, skipping update")
            return
        
        if self.horizon is None:
            smape = 2 * _divide_no_nan(torch.abs(target - preds), torch.abs(target) + torch.abs(preds))
            valid_mask = ~(torch.isnan(smape) | torch.isinf(smape))
            if valid_mask.any():
                self.smape += smape[valid_mask].sum()
                self.nsamples += valid_mask.sum()
        else:
            smape = 2 * _divide_no_nan(torch.abs(target[:, self.horizon] - preds[:, self.horizon]),
                                       torch.abs(target[:, self.horizon]) + torch.abs(preds[:, self.horizon]))
            valid_mask = ~(torch.isnan(smape) | torch.isinf(smape))
            if valid_mask.any():
                self.smape += smape[valid_mask].sum()
                self.nsamples += valid_mask.sum()

    def compute(self):
        if self.nsamples > 0:
            return 100 * (self.smape / self.nsamples)
        else:
            return torch.tensor(float('nan'))


class WAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.FloatTensor([0.0]), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        
        # Check for NaN/Inf
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            warnings.warn("WAPE: NaN or Inf in predictions, skipping update")
            return
        if torch.isnan(target).any() or torch.isinf(target).any():
            warnings.warn("WAPE: NaN or Inf in targets, skipping update")
            return
        
        if self.horizon is None:
            self.numerator += torch.abs(target - preds).sum()
            self.denominator += torch.abs(target).sum()
        else:
            self.numerator += torch.abs(target[:, self.horizon] - preds[:, self.horizon]).sum()
            self.denominator += torch.abs(target[:, self.horizon]).sum()

    def compute(self):
        if self.denominator > 0:
            return 100 * (self.numerator / self.denominator)
        else:
            return torch.tensor(float('nan'))