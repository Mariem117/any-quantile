#!/usr/bin/env python3
"""
Quick test to run just validation and see coverage debug output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from omegaconf import OmegaConf
from model.models import AnyQuantileForecaster
from dataset import ElectricityUnivariateDataModule

def test_validation_only():
    """Test just validation to see coverage debug output"""
    try:
        print("=== Testing Validation Only ===")
        
        # Load config
        with open('config/nbeatsaq-attention-mhlv-fast.yaml') as f:
            cfg_yaml = yaml.unsafe_load(f)
        
        cfg = OmegaConf.create(cfg_yaml)
        
        print("Creating model...")
        model = AnyQuantileForecaster(cfg)
        
        print("Creating datamodule...")
        dm = ElectricityUnivariateDataModule(
            name=cfg.dataset.name,
            num_workers=1,
            train_batch_size=2,
            eval_batch_size=2,
            history_length=cfg.dataset.history_length,
            horizon_length=cfg.dataset.horizon_length,
            split_boundaries=cfg.dataset.split_boundaries,
            fillna=cfg.dataset.fillna,
            train_step=cfg.dataset.train_step,
            eval_step=cfg.dataset.eval_step
        )
        
        dm.prepare_data()
        dm.setup('fit')
        
        print("Getting validation batch...")
        val_loader = dm.val_dataloader()
        batch = next(iter(val_loader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"History shape: {batch['history'].shape}")
        print(f"Target shape: {batch['target'].shape}")
        print(f"Quantiles shape: {batch['quantiles'].shape}")
        print(f"Quantiles: {batch['quantiles'].flatten()}")
        
        # Test validation step
        print("\n--- Running Validation Step ---")
        model.eval()
        with torch.no_grad():
            model.validation_step(batch, 0)
        
        # Test coverage compute
        print("\n--- Computing Coverage ---")
        val_coverage = model.val_coverage.compute()
        print(f"Validation coverage: {val_coverage}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_validation_only()
    if success:
        print("\n=== Validation test completed ===")
    else:
        print("\n=== Validation test failed ===")
