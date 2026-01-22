#!/usr/bin/env python3
"""
Test script to run a minimal training/validation step to see debug coverage output
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_coverage_with_model():
    """Test coverage with actual model components"""
    try:
        import torch
        import yaml
        from omegaconf import OmegaConf
        from model.models import AnyQuantileForecaster
        from metrics import Coverage
        
        print("=== Testing Coverage with Model ===")
        
        # Create a minimal config
        config_dict = {
            'model': {
                'input_horizon_len': 48,
                'max_norm': True,
                'metric_space': 'original',
                'metric_clip': 10000,
                'nn': {
                    'backbone': {
                        '_target_': 'modules.MLP',
                        'layer_width': 64,
                        'num_layers': 2,
                        'size_in': 48,
                        'size_out': 48
                    }
                },
                'optimizer': {
                    '_target_': 'torch.optim.Adam',
                    'lr': 0.001
                },
                'scheduler': None
            }
        }
        
        cfg = OmegaConf.create(config_dict)
        
        # Create model
        print("Creating model...")
        model = AnyQuantileForecaster(cfg)
        
        # Create test batch
        batch_size = 4
        history_len = 48
        horizon_len = 48
        num_quantiles = 3
        
        batch = {
            'history': torch.randn(batch_size, history_len),
            'target': torch.randn(batch_size, horizon_len),
            'quantiles': torch.rand(batch_size, 1) * 0.8 + 0.1  # Random quantiles between 0.1-0.9
        }
        
        print(f"Batch quantiles: {batch['quantiles'].flatten()}")
        
        # Test validation step
        print("\n--- Testing Validation Step ---")
        model.validation_step(batch, 0)
        
        # Test compute
        print("\n--- Testing Coverage Compute ---")
        val_coverage = model.val_coverage.compute()
        print(f"Final validation coverage: {val_coverage}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coverage_with_model()
    if success:
        print("\n=== Test completed successfully ===")
    else:
        print("\n=== Test failed ===")
