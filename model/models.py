import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

# from hydra.utils import instantiate
from utils.model_factory import instantiate

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from metrics import SMAPE, MAPE, CRPS, Coverage
from losses import MonotonicityLoss

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
        
        self.train_mse(y_hat, batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mae(y_hat, batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']
        
        self.val_mse(y_hat, batch['target'])
        self.val_mae(y_hat, batch['target'])
        self.val_smape(y_hat, batch['target'])
                
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
        
        self.test_mse(y_hat, batch['target'])
        self.test_mae(y_hat, batch['target'])
        self.test_smape(y_hat, batch['target'])
        self.test_mape(y_hat, batch['target'])
                
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
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        q = x['quantiles']

        x_max = torch.abs(history).max(dim=-1, keepdims=True)[0]
        if self.cfg.model.max_norm:
            x_max[x_max == 0] = 1
        else:
            # If norm is disabled, set all values to 1
            x_max[x_max >= 0] = 1
        history = history / x_max
        
        forecast = self.backbone(history, q)
        return {'forecast': forecast * x_max[..., None], 'quantiles': q}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']

    def training_step(self, batch, batch_idx):
        # generate random quantiles
        batch_size = batch['history'].shape[0]
        if self.cfg.model.q_sampling == 'fixed_in_batch':
            q = torch.rand(1)
            batch['quantiles'] = (q * torch.ones(batch_size, 1)).to(batch['history'])
        elif self.cfg.model.q_sampling == 'random_in_batch':
            if self.cfg.model.q_distribution == 'uniform':
                batch['quantiles'] = torch.rand(batch_size, 1).to(batch['history'])
            elif self.cfg.model.q_distribution == 'beta':
                batch['quantiles'] = torch.Tensor(np.random.beta(self.cfg.model.q_parameter, self.cfg.model.q_parameter, 
                                                                 size=(batch_size, 1))).to(batch['history'])
            else:
                assert False, f"Option {self.cfg.model.q_distribution} is not implemented for model.q_distribution"
        else:
            assert False, f"Option {self.cfg.model.q_sampling} is not implemented for model.q_sampling"
        
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        center_idx = y_hat.shape[-1]
        assert center_idx % 2 == 1, "Number of quantiles must be odd"
        center_idx = center_idx // 2
        
        loss = self.loss(y_hat, batch['target'], q=quantiles) 
        
        batch_size=batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mse(y_hat[..., center_idx], batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mae(y_hat[..., center_idx], batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch['quantiles'] = self.val_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        
        self.val_mse(y_hat[..., 0].contiguous(), batch['target'])
        self.val_mae(y_hat[..., 0].contiguous(), batch['target'])
        self.val_smape(y_hat[..., 0].contiguous(), batch['target'])
        self.val_mape(y_hat[..., 0].contiguous(), batch['target'])
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
        
        # Update metrics with point forecasts
        self.test_mse(y_hat_point, batch['target'])
        self.test_mae(y_hat_point, batch['target'])
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


class AnyQuantileForecasterWithMonotonicity(AnyQuantileForecaster):
    """Extended forecaster that penalizes quantile crossings."""

    def __init__(self, cfg):
        super().__init__(cfg)
        margin = getattr(cfg.model, "monotone_margin", 0.0)
        self.monotonicity_loss = MonotonicityLoss(margin=margin)
        self.monotone_weight = getattr(cfg.model, "monotone_weight", 0.0)
        self.num_train_quantiles = getattr(cfg.model, "num_train_quantiles", 9)

    def _sample_quantiles(self, batch_size: int, device) -> torch.Tensor:
        num_q = max(1, int(self.num_train_quantiles))
        if num_q % 2 == 0:
            num_q += 1  # keep a clear center quantile for metrics

        q_sampling = getattr(self.cfg.model, "q_sampling", "random_in_batch")
        q_distribution = getattr(self.cfg.model, "q_distribution", "uniform")

        if q_distribution == 'uniform':
            q = torch.rand(1 if q_sampling == 'fixed_in_batch' else batch_size, num_q, device=device)
        elif q_distribution == 'beta':
            beta_param = getattr(self.cfg.model, "q_parameter", 0.3)
            size = (1 if q_sampling == 'fixed_in_batch' else batch_size, num_q)
            q = torch.tensor(np.random.beta(beta_param, beta_param, size=size), device=device, dtype=torch.float32)
        elif q_distribution == 'fixed':
            q_min = getattr(self.cfg.model, "fixed_q_min", 0.1)
            q_max = getattr(self.cfg.model, "fixed_q_max", 0.9)
            q = torch.linspace(q_min, q_max, num_q, device=device)[None]
        else:
            assert False, f"Option {q_distribution} is not implemented for model.q_distribution"

        if q_sampling == 'fixed_in_batch':
            q = q.expand(batch_size, -1)

        q, _ = q.sort(dim=-1)
        return q

    def training_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        batch['quantiles'] = self._sample_quantiles(batch_size, batch['history'].device)

        net_output = self.shared_forward(batch)

        y_hat = net_output['forecast']  # [B, H, Q]
        quantiles = net_output['quantiles']  # [B, Q]
        quantiles_loss = quantiles[:, None]

        pinball_loss = self.loss(y_hat, batch['target'], q=quantiles_loss)
        monotone_loss = self.monotonicity_loss(y_hat, quantiles_loss)
        total_loss = pinball_loss + self.monotone_weight * monotone_loss

        center_idx = y_hat.shape[-1] // 2

        self.log("train/pinball_loss", pinball_loss, on_step=True, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("train/monotone_loss", monotone_loss, on_step=True, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_mse(y_hat[..., center_idx], batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch_size)

        self.train_mae(y_hat[..., center_idx], batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)

        self.train_crps(y_hat, batch['target'], q=quantiles_loss)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=batch_size)

        return total_loss


class AnyQuantileForecasterLog(AnyQuantileForecaster):

    def shared_forward(self, x):
        x['history'] = torch.log(1 + x['history'])
        output = super().shared_forward(x)
        output['forecast_exp'] = torch.exp(output['forecast']) - 1.0
        return output
    
    def training_step(self, batch, batch_idx):
        # generate random quantiles
        batch_size = batch['history'].shape[0]
        if self.cfg.model.q_sampling == 'fixed_in_batch':
            q = torch.rand(1)
            batch['quantiles'] = (q * torch.ones(batch_size, 1)).to(batch['history'])
        elif self.cfg.model.q_sampling == 'random_in_batch':
            batch['quantiles'] = torch.rand(batch_size, 1).to(batch['history'])
        else:
            assert False, f"Option {self.cfg.model.q_sampling} is not implemented for model.q_sampling"
        # batch['quantiles'] = torch.rand(batch['history'].shape[0], 1).to(batch['history'])
        # batch['quantiles'] = (torch.rand(1) * torch.ones(batch['history'].shape[0], 1)).to(batch['history'])
        
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast'] # BxHxQ
        quantiles = net_output['quantiles'][:,None] # Bx1xQ
        y_hat_exp = net_output['forecast_exp'] # BxHxQ
        
        loss = self.loss(y_hat, torch.log(batch['target'] + 1), q=quantiles) 
        
        batch_size=batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mse(y_hat_exp[..., 0], batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mae(y_hat_exp[..., 0], batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat_exp, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return loss
    
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