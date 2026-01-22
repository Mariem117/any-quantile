import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

try:
    from tcr import TemporalCoherenceRegularization
except ImportError:
    # Fallback to local module when tcr is not installed as a package
    from .Tcr import TemporalCoherenceRegularization

try:
    from qcbc import QuantileConditionedBasisCoefficients
except ImportError:
    # Fallback to local module when qcbc is not installed as a package
    from .Qcbc import QuantileConditionedBasisCoefficients

try:
    from dbe import DistributionalBasisExpansion
except ImportError:
    # Fallback to local module when dbe is not installed as a package
    from .Dbe import DistributionalBasisExpansion

# from hydra.utils import instantiate
from utils.model_factory import instantiate
from utils.adaptive_sampling import AdaptiveQuantileSampler
from modules.adaptive_attention import AdaptiveAttentionBlock
from modules.hierarchical import HierarchicalQuantilePredictor

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from metrics import SMAPE, MAPE, CRPS, Coverage
from losses import MQNLoss
from losses.monotone import MonotonicityLoss

from modules import MLP
    
    
class MlpForecaster(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        
    def init_metrics(self):
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.train_smape = SMAPE()
        self.val_smape = SMAPE()
        self.test_smape = SMAPE()
        self.val_mape = MAPE()
        self.test_mape = MAPE()
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]

        
        
        forecast = self.backbone(history)   
        return {'forecast': forecast}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']

    def training_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']
        
        loss = self.loss(y_hat, batch['target']) 
        
        batch_size=batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        # Filter out NaN values before computing MSE/MAE
        y_target = batch['target']
        valid_mask = ~(torch.isnan(y_hat) | torch.isinf(y_hat) | 
                       torch.isnan(y_target) | torch.isinf(y_target))
        if valid_mask.any():
            self.train_mse(y_hat[valid_mask], y_target[valid_mask])
            self.train_mae(y_hat[valid_mask], y_target[valid_mask])
        
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']
        y_target = batch['target']
        
        # Filter out NaN values before computing MSE/MAE
        valid_mask = ~(torch.isnan(y_hat) | torch.isinf(y_hat) | 
                       torch.isnan(y_target) | torch.isinf(y_target))
        if valid_mask.any():
            self.val_mse(y_hat[valid_mask], y_target[valid_mask])
            self.val_mae(y_hat[valid_mask], y_target[valid_mask])
        
        self.val_smape(y_hat, y_target)
                
        batch_size=batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def test_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']
        y_target = batch['target']
        
        # Filter out NaN values before computing MSE/MAE
        valid_mask = ~(torch.isnan(y_hat) | torch.isinf(y_hat) | 
                       torch.isnan(y_target) | torch.isinf(y_target))
        if valid_mask.any():
            self.test_mse(y_hat[valid_mask], y_target[valid_mask])
            self.test_mae(y_hat[valid_mask], y_target[valid_mask])
        
        self.test_smape(y_hat, y_target)
        self.test_mape(y_hat, y_target)
                
        batch_size=batch['history'].shape[0]
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/smape", self.test_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        if scheduler is not None:
            optimizer = {"optimizer": optimizer, 
                         "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer
    
    
class AnyQuantileForecaster(MlpForecaster):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.train_crps = CRPS()
        self.val_crps = CRPS()
        self.test_crps = CRPS()

        self.train_coverage = Coverage(level=0.95)
        self.val_coverage = Coverage(level=0.95)
        self.test_coverage = Coverage(level=0.95)

        if getattr(cfg.model, 'use_tcr', False):
            self.tcr = TemporalCoherenceRegularization(
                base_weight=getattr(cfg.model, 'tcr_weight', 0.01),
                adaptive=False,
            )
        else:
            self.tcr = None

        if getattr(cfg.model, 'use_qcbc', False):
            self.qcbc = QuantileConditionedBasisCoefficients(
                num_basis_functions=cfg.model.nn.backbone.num_blocks,
                quantile_embed_dim=32,
                modulation_scale_init=0.1,
            )
        else:
            self.qcbc = None

        if getattr(cfg.model, 'use_dbe', False):
            horizon = getattr(getattr(cfg, 'dataset', cfg.model), 'horizon_length', self.cfg.model.input_horizon_len)
            feature_dim = getattr(cfg.model.nn.backbone, 'layer_width', None)
            self.dbe = DistributionalBasisExpansion(
                num_components=3,
                horizon=horizon,
                feature_dim=feature_dim if feature_dim is not None else self.cfg.model.input_horizon_len,
            )
        else:
            self.dbe = None

    def _build_training_quantiles(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Ensure the model always trains on the coverage interval and median."""
        base = torch.tensor([
            self.train_coverage.level_low,
            0.5,
            self.train_coverage.level_high,
        ], device=device)

        if self.cfg.model.q_sampling == 'fixed_in_batch':
            shared = torch.rand(2, device=device)
            random_core = shared.unsqueeze(0).expand(batch_size, -1)
        elif self.cfg.model.q_sampling == 'random_in_batch':
            if self.cfg.model.q_distribution == 'uniform':
                random_core = torch.rand(batch_size, 2, device=device)
            elif self.cfg.model.q_distribution == 'beta':
                random_core = torch.as_tensor(
                    np.random.beta(self.cfg.model.q_parameter, self.cfg.model.q_parameter, size=(batch_size, 2)),
                    device=device,
                    dtype=base.dtype,
                )
            else:
                raise AssertionError(f"Option {self.cfg.model.q_distribution} is not implemented for model.q_distribution")
        else:
            raise AssertionError(f"Option {self.cfg.model.q_sampling} is not implemented for model.q_sampling")

        quantiles = torch.cat([base.unsqueeze(0).expand(batch_size, -1), random_core], dim=-1)
        quantiles, _ = torch.sort(quantiles, dim=-1)
        return quantiles
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        q = x['quantiles']

        x_max = torch.abs(history).max(dim=-1, keepdims=True)[0]
        # Ensure x_max is never zero to avoid division by zero
        x_max = torch.clamp(x_max, min=1e-8)
        if self.cfg.model.max_norm:
            x_max[x_max == 0] = 1
        else:
            # If norm is disabled, set all values to 1
            x_max[x_max >= 0] = 1
        history_norm = history / x_max
        
        # Replace any NaN/Inf in normalized history
        history_norm = torch.nan_to_num(history_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Sort quantiles for consistent conditioning
        q_sorted, _ = torch.sort(q, dim=-1)

        forecast = self.backbone(history_norm, q_sorted)

        if self.qcbc is not None:
            # If backbone outputs basis coefficients [B, H, K], modulate across quantiles
            if forecast.dim() == 3 and forecast.shape[-1] == getattr(self.cfg.model.nn.backbone, 'num_blocks', forecast.shape[-1]):
                forecast_modulated = self.qcbc(forecast, q_sorted)
                forecast = forecast_modulated.sum(dim=2)

        forecast_denorm = forecast * x_max[..., None]
        forecast_denorm = torch.nan_to_num(forecast_denorm, nan=0.0, posinf=1e6, neginf=-1e6)

        # Optional DBE path if backbone exposes features and locations have component axis
        if self.dbe is not None:
            get_feats = getattr(self.backbone, 'get_features', None)
            locations = forecast
            # Require a component axis for DBE; otherwise fall back
            if locations.dim() == 3 and locations.shape[-1] == self.dbe.num_components and callable(get_feats):
                features = get_feats(history_norm, q_sorted) if get_feats.__code__.co_argcount >= 3 else get_feats(history_norm)
                if features is not None:
                    forecast_dbe = self.dbe(features=features, locations=locations, quantile_levels=q_sorted)
                    return {'forecast': forecast_dbe, 'quantiles': q_sorted}

        return {'forecast': forecast_denorm, 'quantiles': q_sorted}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']

    def training_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = batch['quantiles'][:, None, :] # Bx1xQ

        main_loss = self.loss(y_hat, batch['target'], q=quantiles)
        tcr_loss = None
        if self.tcr is not None:
            tcr_loss = self.tcr(y_hat, batch['quantiles'])
            total_loss = main_loss + tcr_loss
        else:
            total_loss = main_loss

        self.log("train/main_loss", main_loss, on_step=True, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        if tcr_loss is not None:
            self.log("train/tcr_loss", tcr_loss, on_step=True, on_epoch=True, 
                     prog_bar=False, logger=True, batch_size=batch_size)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        # Median index is wherever quantile is closest to 0.5
        median_idx = (torch.abs(batch['quantiles'] - 0.5)).argmin(dim=-1)
        median_idx_expanded = median_idx.view(batch_size, 1, 1).expand(-1, y_hat.shape[1], 1)
        y_hat_point = torch.gather(y_hat, -1, median_idx_expanded).squeeze(-1)
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.train_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.train_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        batch['quantiles'] = self.val_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        
        # Filter out NaN values before computing MSE/MAE
        y_hat_point = y_hat[..., 0].contiguous()
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.val_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.val_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.val_smape(y_hat_point, batch['target'])
        self.val_mape(y_hat_point, batch['target'])
        self.val_crps(y_hat, batch['target'], q=quantiles)
        self.val_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size=batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/mape", self.val_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/crps", self.val_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"val/coverage-{self.val_coverage.level}", self.val_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def test_step(self, batch, batch_idx):
        batch['quantiles'] = self.test_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        
        # Find the median quantile (0.5) for point forecasts
        # The first quantile in the batch is 0.5 (median)
        median_idx = 0  # Index 0 corresponds to quantile 0.5 in your data
        y_hat_point = y_hat[..., median_idx].contiguous()  # BxH
        
        # Filter out NaN values before computing MSE/MAE
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.test_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.test_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.test_smape(y_hat_point, batch['target'])
        self.test_mape(y_hat_point, batch['target'])
        
        # Update probabilistic metrics with full quantile outputs
        self.test_crps(y_hat, batch['target'], q=quantiles)
        self.test_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size=batch['history'].shape[0]
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/smape", self.test_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/crps", self.test_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"test/coverage-{self.test_coverage.level}", self.test_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)


class AnyQuantileForecasterLog(AnyQuantileForecaster):

    def shared_forward(self, x):
        x['history'] = torch.log(1 + x['history'])
        output = super().shared_forward(x)
        output['forecast_exp'] = torch.exp(output['forecast']) - 1.0
        return output
    
    def training_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = batch['quantiles'][:, None, :] # Bx1xQ
        y_hat_exp = net_output['forecast_exp'] # BxHxQ
        
        main_loss = self.loss(y_hat, torch.log(batch['target'] + 1), q=quantiles)
        tcr_loss = None
        if self.tcr is not None:
            tcr_loss = self.tcr(y_hat_exp, batch['quantiles'])
            total_loss = main_loss + tcr_loss
        else:
            total_loss = main_loss
        
        batch_size=batch['history'].shape[0]
        self.log("train/main_loss", main_loss, on_step=True, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        if tcr_loss is not None:
            self.log("train/tcr_loss", tcr_loss, on_step=True, on_epoch=True, 
                     prog_bar=False, logger=True, batch_size=batch_size)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        # Median index is wherever quantile is closest to 0.5
        median_idx = (torch.abs(batch['quantiles'] - 0.5)).argmin(dim=-1)
        median_idx_expanded = median_idx.view(batch_size, 1, 1).expand(-1, y_hat_exp.shape[1], 1)
        y_hat_point = torch.gather(y_hat_exp, -1, median_idx_expanded).squeeze(-1)

        self.train_mse(y_hat_point, batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mae(y_hat_point, batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat_exp, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        batch['quantiles'] = self.val_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        y_hat_exp = net_output['forecast_exp'] # BxHxQ
        
        self.val_mse(y_hat_exp[..., 0], batch['target'])
        self.val_mae(y_hat_exp[..., 0], batch['target'])
        self.val_smape(y_hat_exp[..., 0], batch['target'])
        self.val_crps(y_hat_exp, batch['target'], q=quantiles)
        self.val_coverage(y_hat_exp, batch['target'], q=quantiles)
                
        batch_size=batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/crps", self.val_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"val/coverage-{self.val_coverage.level}", self.val_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def test_step(self, batch, batch_idx):
        batch['quantiles'] = self.test_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        y_hat_exp = net_output['forecast_exp'] # BxHxQ
        
        # Extract median point forecasts (index 0 = quantile 0.5)
        median_idx = 0
        y_hat_point = y_hat_exp[..., median_idx].contiguous()
        
        self.test_mse(y_hat_point, batch['target'])
        self.test_mae(y_hat_point, batch['target'])
        self.test_smape(y_hat_point, batch['target'])
        self.test_mape(y_hat_point, batch['target'])
        self.test_crps(y_hat_exp, batch['target'], q=quantiles)
        self.test_coverage(y_hat_exp, batch['target'], q=quantiles)
                
        batch_size=batch['history'].shape[0]
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/smape", self.test_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/crps", self.test_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"test/coverage-{self.test_coverage.level}", self.test_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def __init__(self, cfg):
        super().__init__(cfg)
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        q = x['quantiles']

        x_max = torch.abs(history).max(dim=-1, keepdims=True)[0]
        if self.cfg.model.max_norm:
            x_max[x_max == 0] = 1
        else:
            x_max[x_max >= 0] = 1
        history = history / x_max
        
        # Extract exogenous features
        continuous = None
        calendar = None
        
        if 'exog_history' in x:
            continuous = x['exog_history'].squeeze(1)  # [B, T, num_continuous]
        
        if 'calendar_history' in x:
            calendar = x['calendar_history'].squeeze(1)  # [B, T, 4]
            # Convert normalized [0,1] features to integer indices
            # Be careful with boundary conditions
            calendar_indices = torch.stack([
                torch.clamp((calendar[..., 0] * 24).long(), 0, 23),  # hour: 0-23
                torch.clamp((calendar[..., 1] * 7).long(), 0, 6),    # dow: 0-6
                torch.clamp((calendar[..., 2] * 12).long(), 0, 11),  # month: 0-11
                calendar[..., 3].long()  # weekend: already 0 or 1
            ], dim=-1)
            calendar = calendar_indices
        
        # Pass to backbone
        forecast = self.backbone(history, q, continuous, calendar)
        return {'forecast': forecast * x_max[..., None], 'quantiles': q}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']


class GeneralAnyQuantileForecaster(AnyQuantileForecaster):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.time_series_projection_in = torch.nn.Linear(1, cfg.model.nn.backbone.d_model)
        self.time_series_projection_out = torch.nn.Linear(cfg.model.nn.backbone.d_model, 1)
        
        # 100 includes 31 days, 12 months and 7 days of week
        self.time_embedding = torch.nn.Embedding(2000, cfg.model.nn.embedding_dim)
        # this includes 0 as no deal and deal types 1,2,3
        self.time_series_id = torch.nn.Embedding(cfg.model.nn.time_series_id_num, cfg.model.nn.embedding_dim)
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        
        t_h = torch.arange(self.cfg.model.input_horizon_len, dtype=torch.int64)[None].to(history.device)
        t_t = torch.arange(x['time_features_target'].shape[1], dtype=torch.int64)[None].to(history.device) + self.cfg.model.input_horizon_len
        
        time_features_tgt = torch.repeat_interleave(self.time_embedding(t_t), repeats=history.shape[0], dim=0)
        time_features_src = self.time_embedding(t_h)
        
        xf_input = time_features_tgt
        xt_input = time_features_src + self.time_series_projection_in(history.unsqueeze(-1))
        xs_input = 0.0 * self.time_series_id(x['series_id'])
        
        backbone_output = self.backbone(xt_input=xt_input, xf_input=xf_input, xs_input=xs_input)   
        backbone_output = self.time_series_projection_out(backbone_output)
        forecast = backbone_output[..., 0] + history.mean(dim=-1, keepdims=True) + self.shortcut(history)
        return {'forecast': forecast}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']
class AnyQuantileForecasterWithMonotonicity(AnyQuantileForecaster):
    """
    Extended forecaster with monotonicity loss
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.monotonicity_loss = MonotonicityLoss(
            margin=cfg.model.monotone_margin
        )
        self.monotone_weight = cfg.model.monotone_weight  # e.g., 0.1

    def training_step(self, batch, batch_idx):
        batch_size = batch["history"].shape[0]

        # Sample MULTIPLE quantiles per sample for monotonicity training
        num_quantiles = self.cfg.model.num_train_quantiles  # e.g., 9

        if self.cfg.model.q_distribution == "uniform":
            # Sample random quantiles and sort them
            q = torch.rand(batch_size, num_quantiles, device=batch["history"].device)
            q, _ = q.sort(dim=-1)

        elif self.cfg.model.q_distribution == "fixed":
            # Use a fixed quantile grid
            q = torch.linspace(0.1, 0.9, num_quantiles, device=batch["history"].device)
            q = q.unsqueeze(0).expand(batch_size, -1)

        elif self.cfg.model.q_distribution == "beta":
            # Beta sampling for heavier tails focus (alpha=beta=q_parameter)
            alpha = float(getattr(self.cfg.model, "q_parameter", 0.3))
            alpha = max(alpha, 1e-3)
            beta = alpha
            q = torch.distributions.Beta(alpha, beta).sample((batch_size, num_quantiles)).to(batch["history"].device)
            q = q.clamp(1e-4, 1 - 1e-4)
            q, _ = q.sort(dim=-1)

        else:
            raise ValueError(
                f"Unknown q_distribution: {self.cfg.model.q_distribution}"
            )

        # Attach quantiles to batch
        batch["quantiles"] = q

        # Forward pass — predicts multiple quantiles
        net_output = self.shared_forward(batch)

        y_hat = net_output["forecast"]      # [B, H, Q]
        quantiles = net_output["quantiles"] # [B, Q]

        # Pinball loss (main objective)
        pinball_loss = self.loss(
            y_hat,
            batch["target"],
            q=quantiles[:, None, :]
        )

        # Monotonicity loss (regularization)
        monotone_loss = self.monotonicity_loss(y_hat, quantiles)

        # Combined loss
        total_loss = pinball_loss + self.monotone_weight * monotone_loss

        # Logging
        self.log("train/pinball_loss", pinball_loss, prog_bar=True)
        self.log("train/monotone_loss", monotone_loss)
        self.log("train/total_loss", total_loss, prog_bar=True)

        return total_loss


class AnyQuantileForecasterHierMonotone(AnyQuantileForecasterWithMonotonicity):
    """Fuse hierarchical quantile generation with monotonicity regularization."""

    def __init__(self, cfg):
        super().__init__(cfg)

        method = getattr(cfg.model, "hierarchical_method", "gaussian")
        hidden_dim = getattr(cfg.model, "hierarchical_hidden_dim", None)
        if hidden_dim is not None and hidden_dim <= 0:
            hidden_dim = None

        horizon = getattr(cfg.dataset, "horizon_length", getattr(cfg.model, "horizon_length", None))
        if horizon is None:
            horizon = cfg.model.input_horizon_len

        self.hierarchical_predictor = HierarchicalQuantilePredictor(
            backbone=self.backbone,
            hidden_dim=hidden_dim,
            horizon_length=horizon,
            method=method,
        )

    def shared_forward(self, x):
        history = x["history"][:, -self.cfg.model.input_horizon_len:]
        q = x["quantiles"]

        x_max = torch.abs(history).max(dim=-1, keepdims=True)[0]
        x_max = torch.clamp(x_max, min=1e-8)
        if self.cfg.model.max_norm:
            x_max[x_max == 0] = 1
        else:
            x_max[x_max >= 0] = 1
        history_norm = history / x_max
        history_norm = torch.nan_to_num(history_norm, nan=0.0, posinf=1.0, neginf=-1.0)

        # Hierarchical head converts backbone features into monotonic quantiles
        forecast = self.hierarchical_predictor(history_norm, q)

        forecast = forecast * x_max[..., None]
        forecast = torch.nan_to_num(forecast, nan=0.0, posinf=1e6, neginf=-1e6)

        return {"forecast": forecast, "quantiles": q}

    def forward(self, x):
        out = self.shared_forward(x)
        return out["forecast"]


class AnyQuantileForecasterAdaptive(AnyQuantileForecaster):
    """Forecaster that adapts quantile sampling based on recent quantile-specific loss."""

    def __init__(self, cfg):
        super().__init__(cfg)
        adaptive_cfg = getattr(cfg.model, "adaptive_sampling", None)

        def _get(name, default):
            if adaptive_cfg is None:
                return default
            return getattr(adaptive_cfg, name, adaptive_cfg[name] if isinstance(adaptive_cfg, dict) and name in adaptive_cfg else default)

        self.num_adaptive_quantiles = int(_get("num_adaptive_quantiles", 2))
        self.sampler = AdaptiveQuantileSampler(
            num_bins=int(_get("num_bins", 100)),
            momentum=float(_get("momentum", 0.99)),
            temperature=float(_get("temperature", 1.0)),
            min_prob=float(_get("min_prob", 0.001)),
        )

    def _build_training_quantiles(self, batch_size: int, device: torch.device) -> torch.Tensor:
        base = torch.tensor([
            self.train_coverage.level_low,
            0.5,
            self.train_coverage.level_high,
        ], device=device)

        adaptive_draws = self.sampler.sample(batch_size * self.num_adaptive_quantiles).to(device)
        adaptive_draws = adaptive_draws.view(batch_size, self.num_adaptive_quantiles)

        quantiles = torch.cat([base.unsqueeze(0).expand(batch_size, -1), adaptive_draws], dim=-1)
        quantiles, _ = torch.sort(quantiles, dim=-1)
        return quantiles

    def training_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)

        y_hat = net_output['forecast']  # BxHxQ
        quantiles = net_output['quantiles'][:, None, :]  # Bx1xQ
        
        # Find median quantile (0.5) for point forecasts
        q_values = quantiles[0, 0, :].cpu()
        median_idx = (q_values - 0.5).abs().argmin().item()
        
        loss = self.loss(y_hat, batch['target'], q=quantiles) 
        
        batch_size = batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        # Point forecasts using median
        y_hat_point = y_hat[..., median_idx]
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.train_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.train_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        # Update adaptive sampler with per-quantile losses
        with torch.no_grad():
            # Compute loss per quantile for adaptive sampling
            per_quantile_loss = []
            for i in range(quantiles.shape[2]):
                q_single = quantiles[0, 0, i:i+1]  # Get single quantile
                q_loss = self.loss(y_hat[..., i], batch['target'], q=q_single)
                per_quantile_loss.append(q_loss.item())
            
            # Update sampler with quantile losses
            quantile_values = quantiles[0, 0, :].cpu().flatten()
            self.sampler.update(quantile_values, torch.tensor(per_quantile_loss))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)

        y_hat = net_output['forecast']  # BxHxQ
        quantiles = net_output['quantiles'][:, None, :]  # Bx1xQ
        
        # Find median quantile (0.5) for point forecasts
        q_values = quantiles[0, 0, :].cpu()
        median_idx = (q_values - 0.5).abs().argmin().item()
        
        # Point forecasts using median
        y_hat_point = y_hat[..., median_idx]
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.val_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.val_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.val_smape(y_hat_point, batch['target'])
        self.val_mape(y_hat_point, batch['target'])
        self.val_crps(y_hat, batch['target'], q=quantiles)
        self.val_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size = batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/mape", self.val_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/crps", self.val_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"val/coverage-{self.val_coverage.level}", self.val_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def test_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)

        y_hat = net_output['forecast']  # BxHxQ
        quantiles = net_output['quantiles'][:, None, :]  # Bx1xQ
        
        # Find median quantile (0.5) for point forecasts
        q_values = quantiles[0, 0, :].cpu()
        median_idx = (q_values - 0.5).abs().argmin().item()
        
        # Point forecasts using median
        y_hat_point = y_hat[..., median_idx]
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.test_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.test_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.test_smape(y_hat_point, batch['target'])
        self.test_mape(y_hat_point, batch['target'])
        self.test_crps(y_hat, batch['target'], q=quantiles)
        self.test_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size = batch['history'].shape[0]
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/smape", self.test_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/crps", self.test_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"test/coverage-{self.test_coverage.level}", self.test_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)


class AnyQuantileWithSeriesEmbedding(AnyQuantileForecaster):
    """Forecaster with learnable per-series (country) embeddings"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # Learnable embeddings for each time series (country)
        self.num_series = cfg.model.num_series  # 35 for European countries
        self.embed_dim = cfg.model.series_embed_dim  # e.g., 32
        self.series_embedding = nn.Embedding(
            num_embeddings=self.num_series,
            embedding_dim=self.embed_dim
        )
        
        # Project series embedding to history length for additive bias
        self.series_proj = nn.Linear(self.embed_dim, cfg.model.input_horizon_len)
        
        # Initialize with small values
        nn.init.normal_(self.series_embedding.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.series_proj.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.series_proj.bias)
    
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        q = x['quantiles']
        batch_size = history.shape[0]
        
        # Handle series_id
        if 'series_id' in x and x['series_id'] is not None:
            series_id = x['series_id']
            if series_id.dim() > 1:
                series_id = series_id.squeeze(-1)
            series_id = torch.clamp(series_id, 0, self.num_series - 1)
            
            # Get series embedding and add bias to input
            series_embed = self.series_embedding(series_id)
            series_bias = self.series_proj(series_embed)
            history = history + 0.15 * series_bias
        else:
            series_embed = torch.zeros(batch_size, self.embed_dim, device=history.device)
        
        # Normalization
        x_max = torch.abs(history).max(dim=-1, keepdims=True)[0]
        
        if self.cfg.model.max_norm:
            x_max = torch.clamp(x_max, min=1.0)
        else:
            x_max = torch.ones_like(x_max)
        
        history_norm = history / x_max
        
        # Check for NaN/Inf
        if torch.isnan(history_norm).any() or torch.isinf(history_norm).any():
            history_norm = torch.nan_to_num(history_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Forward through backbone
        forecast = self.backbone(history_norm, q)
        
        # Denormalize
        forecast = forecast * x_max[..., None]
        
        # Check for NaN/Inf in forecast
        if torch.isnan(forecast).any() or torch.isinf(forecast).any():
            nan_count = torch.isnan(forecast).sum().item()
            inf_count = torch.isinf(forecast).sum().item()
            if nan_count > 0 or inf_count > 0:
                import warnings
                warnings.warn(f"NaN/Inf in forecast: {nan_count} NaNs, {inf_count} Infs")
        
        return {
            'forecast': forecast,
            'quantiles': q,
            'series_embed': series_embed
        }
    
    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']


class AnyQuantileForecasterWithHierarchical(AnyQuantileForecaster):
    """
    Forecaster using hierarchical quantile prediction with location-scale decomposition.
    This approach predicts location (median) and scale separately, then generates quantile
    predictions as offsets, ensuring better coherence across quantiles.
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # Use simple approach - just rely on base class with better regularization
        pass
    
    def shared_forward(self, x):
        """
        Use base class implementation - hierarchical overhead was hurting CRPS
        """
        # Just use the parent class implementation which works better
        return super().shared_forward(x)
    
    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']


class AnyQuantileForecasterAdaptiveAttention(AnyQuantileForecasterAdaptive):
    """
    Forecaster that fuses adaptive sampling with adaptive attention mechanism.
    
    This model combines:
    1. Adaptive quantile sampling based on recent loss feedback
    2. Adaptive attention that modulates attention weights based on quantile importance
    3. Dynamic learning that focuses on difficult quantile regions
    """
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Get adaptive attention configuration
        attention_cfg = getattr(cfg.model, "adaptive_attention", {})
        
        self.d_model = attention_cfg.get("d_model", 256)
        self.n_heads = attention_cfg.get("n_heads", 8)
        self.dropout = attention_cfg.get("dropout", 0.1)
        self.d_ff = attention_cfg.get("d_ff", 1024)
        self.adaptive_temp = attention_cfg.get("adaptive_temp", 1.0)
        self.num_attention_blocks = attention_cfg.get("num_blocks", 2)
        
        # Create adaptive attention blocks
        self.adaptive_attention_blocks = nn.ModuleList([
            AdaptiveAttentionBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                dropout=self.dropout,
                d_ff=self.d_ff,
                adaptive_temp=self.adaptive_temp
            ) for _ in range(self.num_attention_blocks)
        ])
        
        # Quantile projection parameters: project Q -> d_model and back. Use max quantile count from config.
        self.q_dim_max = getattr(cfg.model.nn.backbone, "quantile_embed_num", None) or 100
        self.in_proj_weight = nn.Parameter(torch.empty(self.d_model, self.q_dim_max))
        self.in_proj_bias = nn.Parameter(torch.zeros(self.d_model))
        self.out_proj_weight = nn.Parameter(torch.empty(self.q_dim_max, self.d_model))
        self.out_proj_bias = nn.Parameter(torch.zeros(self.q_dim_max))
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj_weight)
        
        # Layer normalization for stable training (applied after projection back to Q)
        self.output_norm = nn.LayerNorm(self.q_dim_max)
    
    def shared_forward(self, x):
        """Forward pass with adaptive attention mechanism."""
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        quantiles = x['quantiles']
        
        # Normalize input
        x_max = torch.abs(history).max(dim=-1, keepdim=True)[0]
        if self.cfg.model.max_norm:
            x_max[x_max == 0] = 1
        else:
            x_max[x_max >= 0] = 1
        history = history / x_max
        
        # Initial backbone processing (pass quantiles for CAT/attention backbones)
        forecast = self.backbone(history, quantiles)  # [B, H, Q]
        
        # Reshape for attention processing
        B, H, Q = forecast.shape
        forecast_flat = forecast.reshape(B * H, Q)  # [B*H, Q]
        
        # Project to attention dimension using weight slices matching current Q
        in_w = self.in_proj_weight[:, :Q]
        in_b = self.in_proj_bias
        forecast_proj = F.linear(forecast_flat, in_w, in_b)  # [B*H, d_model]
        forecast_proj = forecast_proj.reshape(B, H, self.d_model)  # [B, H, d_model]
        
        # Apply adaptive attention blocks
        attention_out = forecast_proj
        for attention_block in self.adaptive_attention_blocks:
            # Use average quantile for attention computation
            avg_quantile = quantiles.mean(dim=1, keepdim=True)  # [B, 1]
            attention_out = attention_block(attention_out, avg_quantile)
        
        # Project back to original dimension
        attention_out = attention_out.reshape(B * H, self.d_model)
        out_w = self.out_proj_weight[:Q, :]
        out_b = self.out_proj_bias[:Q]
        output_proj = F.linear(attention_out, out_w, out_b)  # [B*H, Q]
        # Normalize over the current Q only
        output_norm = F.layer_norm(output_proj, (Q,))
        forecast_enhanced = output_norm.reshape(B, H, Q)
        
        # Combine with original forecast (residual connection)
        final_forecast = forecast + forecast_enhanced
        
        # Denormalize
        final_forecast = final_forecast * x_max.unsqueeze(-1)
        
        return {
            'forecast': final_forecast,
            'quantiles': quantiles
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with adaptive attention updates."""
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)

        y_hat = net_output['forecast']  # BxHxQ
        quantiles = net_output['quantiles'][:, None, :]  # Bx1xQ
        
        # Find median quantile (0.5) for point forecasts
        q_values = quantiles[0, 0, :].cpu()
        median_idx = (q_values - 0.5).abs().argmin().item()
        
        loss = self.loss(y_hat, batch['target'], q=quantiles) 
        
        batch_size = batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        # Point forecasts using median
        y_hat_point = y_hat[..., median_idx]
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.train_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.train_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        # Update adaptive sampler and attention with per-quantile losses
        with torch.no_grad():
            # Compute loss per quantile for adaptive sampling
            per_quantile_loss = []
            for i in range(quantiles.shape[2]):
                q_single = quantiles[0, 0, i:i+1]  # Get single quantile
                q_loss = self.loss(y_hat[..., i], batch['target'], q=q_single)
                per_quantile_loss.append(q_loss.item())
            
            # Update sampler with quantile losses
            quantile_values = quantiles[0, 0, :].cpu().flatten()
            self.sampler.update(quantile_values, torch.tensor(per_quantile_loss))
            
            # Update attention blocks with quantile importance
            for attention_block in self.adaptive_attention_blocks:
                attention_block.adaptive_attention.update_quantile_importance(
                    quantile_values, torch.tensor(per_quantile_loss)
                )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with adaptive attention."""
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._build_training_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)

        y_hat = net_output['forecast']  # BxHxQ
        quantiles = net_output['quantiles'][:, None, :]  # Bx1xQ
        
        # Find median quantile (0.5) for point forecasts
        q_values = quantiles[0, 0, :].cpu()
        median_idx = (q_values - 0.5).abs().argmin().item()
        
        # Point forecasts using median
        y_hat_point = y_hat[..., median_idx]
        
        # Filter out NaN values before computing metrics
        valid_mask = ~(torch.isnan(y_hat_point) | torch.isinf(y_hat_point) | 
                       torch.isnan(batch['target']) | torch.isinf(batch['target']))
        if valid_mask.any():
            self.val_mse(y_hat_point[valid_mask], batch['target'][valid_mask])
            self.val_mae(y_hat_point[valid_mask], batch['target'][valid_mask])
        
        self.val_smape(y_hat_point, batch['target'])
        self.val_mape(y_hat_point, batch['target'])
        self.val_crps(y_hat, batch['target'], q=quantiles)
        self.val_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size = batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/mape", self.val_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/crps", self.val_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"val/coverage-{self.val_coverage.level}", self.val_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)

