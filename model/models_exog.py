import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.model_factory import instantiate

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from metrics import SMAPE, MAPE, CRPS, Coverage

from modules import NBEATSEXOG


class AnyQuantileForecasterExog(pl.LightningModule):
    """AnyQuantile forecaster with exogenous feature support"""
    
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
        
        # Probabilistic metrics
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
        
        # Extract exogenous features if available
        continuous = None
        calendar = None
        
        if 'exog_history' in x:
            continuous = x['exog_history'].squeeze(1)  # [B, T, num_continuous]
        
        if 'calendar_history' in x:
            calendar = x['calendar_history'].squeeze(1)  # [B, T, 4]
            # Convert normalized calendar features back to indices for embeddings
            calendar_indices = torch.stack([
                (calendar[..., 0] * 23).long().clamp(0, 23),  # hour 0-23
                (calendar[..., 1] * 6).long().clamp(0, 6),    # day of week 0-6
                (calendar[..., 2] * 11).long().clamp(0, 11),  # month 0-11
                calendar[..., 3].long().clamp(0, 1)           # weekend 0-1
            ], dim=-1)
            calendar = calendar_indices
        
        forecast = self.backbone(history, q, continuous, calendar)
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

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        if scheduler is not None:
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "interval": "step",
                    "frequency": 1,
                    "strict": False  # This prevents the warning
                }
            }
        return optimizer