#!/usr/bin/env python3
"""
Test script for the fused adaptive sampling + attention model.

This script tests the AnyQuantileForecasterAdaptiveAttention model which combines:
1. Adaptive quantile sampling based on recent loss feedback
2. Adaptive attention that modulates attention weights based on quantile importance
3. Dynamic learning that focuses on difficult quantile regions
"""

import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Import the new model and related components
from model import AnyQuantileForecasterAdaptiveAttention
from modules import AdaptiveAttention, AdaptiveAttentionBlock
from utils.adaptive_sampling import AdaptiveQuantileSampler


def create_test_config():
    """Create a minimal configuration for testing."""
    cfg = OmegaConf.create({
        'model': {
            'input_horizon_len': 24,
            'nn': {
                'backbone': {
                    '_target_': 'modules.NBEATSAQCAT',
                    'width': 256,
                    'layers': 3,
                    'blocks': 30
                }
            },
            'loss': {
                '_target_': 'losses.MQNLoss'
            },
            'max_norm': True,
            'adaptive_sampling': {
                'num_adaptive_quantiles': 2,
                'num_bins': 100,
                'momentum': 0.99,
                'temperature': 1.0,
                'min_prob': 0.001
            },
            'adaptive_attention': {
                'd_model': 256,
                'n_heads': 8,
                'dropout': 0.1,
                'd_ff': 1024,
                'adaptive_temp': 1.0,
                'num_blocks': 2
            }
        },
        'data': {
            'target_scaler': {
                '_target_': 'utils.scalers.StandardScaler'
            }
        }
    })
    return cfg


def create_test_batch(batch_size=4, history_len=168, forecast_len=24, num_features=1):
    """Create synthetic batch data for testing."""
    batch = {
        'history': torch.randn(batch_size, history_len, num_features),
        'target': torch.randn(batch_size, forecast_len, num_features),
        'quantiles': torch.tensor([[0.1, 0.5, 0.9]]).repeat(batch_size, 1)
    }
    return batch


def test_adaptive_attention_module():
    """Test the AdaptiveAttention module independently."""
    print("Testing AdaptiveAttention module...")
    
    # Create test data
    batch_size, seq_len, d_model = 4, 24, 256
    x = torch.randn(batch_size, seq_len, d_model)
    quantile = torch.tensor([[0.5], [0.1], [0.9], [0.7]])
    
    # Initialize adaptive attention
    adaptive_attention = AdaptiveAttention(
        d_model=d_model,
        n_heads=8,
        dropout=0.1
    )
    
    # Test forward pass
    try:
        output = adaptive_attention(x, quantile)
        assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
        print("✓ AdaptiveAttention forward pass successful")
    except Exception as e:
        print(f"✗ AdaptiveAttention forward pass failed: {e}")
        return False
    
    # Test importance update
    try:
        quantiles = torch.tensor([0.1, 0.5, 0.9])
        losses = torch.tensor([0.8, 0.3, 0.6])
        adaptive_attention.update_quantile_importance(quantiles, losses)
        print("✓ AdaptiveAttention importance update successful")
    except Exception as e:
        print(f"✗ AdaptiveAttention importance update failed: {e}")
        return False
    
    return True


def test_adaptive_attention_block():
    """Test the AdaptiveAttentionBlock."""
    print("\nTesting AdaptiveAttentionBlock...")
    
    # Create test data
    batch_size, seq_len, d_model = 4, 24, 256
    x = torch.randn(batch_size, seq_len, d_model)
    quantile = torch.tensor([[0.5], [0.1], [0.9], [0.7]])
    
    # Initialize block
    block = AdaptiveAttentionBlock(
        d_model=d_model,
        n_heads=8,
        dropout=0.1
    )
    
    try:
        output = block(x, quantile)
        assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
        print("✓ AdaptiveAttentionBlock forward pass successful")
        return True
    except Exception as e:
        print(f"✗ AdaptiveAttentionBlock forward pass failed: {e}")
        return False


def test_adaptive_quantile_sampler():
    """Test the AdaptiveQuantileSampler."""
    print("\nTesting AdaptiveQuantileSampler...")
    
    sampler = AdaptiveQuantileSampler(num_bins=10, momentum=0.9)
    
    # Test sampling
    try:
        samples = sampler.sample(batch_size=4)
        assert samples.shape == (4,), f"Expected shape (4,), got {samples.shape}"
        assert torch.all(samples >= 0) and torch.all(samples <= 1), "Samples should be in [0, 1]"
        print("✓ AdaptiveQuantileSampler sampling successful")
    except Exception as e:
        print(f"✗ AdaptiveQuantileSampler sampling failed: {e}")
        return False
    
    # Test update
    try:
        quantiles = torch.tensor([0.1, 0.5, 0.9])
        losses = torch.tensor([0.8, 0.3, 0.6])
        sampler.update(quantiles, losses)
        print("✓ AdaptiveQuantileSampler update successful")
    except Exception as e:
        print(f"✗ AdaptiveQuantileSampler update failed: {e}")
        return False
    
    return True


def test_fused_model():
    """Test the complete fused model."""
    print("\nTesting AnyQuantileForecasterAdaptiveAttention...")
    
    # Create configuration and model
    cfg = create_test_config()
    
    try:
        model = AnyQuantileForecasterAdaptiveAttention(cfg)
        print("✓ Model initialization successful")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False
    
    # Create test batch
    batch = create_test_batch()
    
    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model.shared_forward(batch)
        
        assert 'forecast' in output, "Output should contain 'forecast' key"
        assert 'quantiles' in output, "Output should contain 'quantiles' key"
        
        forecast_shape = output['forecast'].shape
        expected_shape = (batch['history'].shape[0], 24, batch['quantiles'].shape[1])
        assert forecast_shape == expected_shape, f"Expected forecast shape {expected_shape}, got {forecast_shape}"
        
        print("✓ Model forward pass successful")
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False
    
    # Test training step
    try:
        model.train()
        loss = model.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor), "Training step should return a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        print("✓ Model training step successful")
    except Exception as e:
        print(f"✗ Model training step failed: {e}")
        return False
    
    # Test validation step
    try:
        model.eval()
        model.validation_step(batch, batch_idx=0)
        print("✓ Model validation step successful")
    except Exception as e:
        print(f"✗ Model validation step failed: {e}")
        return False
    
    return True


def test_model_with_different_quantiles():
    """Test model behavior with different quantile configurations."""
    print("\nTesting model with different quantile configurations...")
    
    cfg = create_test_config()
    model = AnyQuantileForecasterAdaptiveAttention(cfg)
    
    # Test with single quantile
    batch_single = create_test_batch()
    batch_single['quantiles'] = torch.tensor([[0.5]]).repeat(4, 1)
    
    try:
        model.eval()
        with torch.no_grad():
            output = model.shared_forward(batch_single)
        assert output['forecast'].shape[2] == 1, "Should produce single quantile output"
        print("✓ Single quantile test successful")
    except Exception as e:
        print(f"✗ Single quantile test failed: {e}")
        return False
    
    # Test with multiple quantiles
    batch_multi = create_test_batch()
    batch_multi['quantiles'] = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]]).repeat(4, 1)
    
    try:
        model.eval()
        with torch.no_grad():
            output = model.shared_forward(batch_multi)
        assert output['forecast'].shape[2] == 5, "Should produce 5 quantile outputs"
        print("✓ Multiple quantiles test successful")
    except Exception as e:
        print(f"✗ Multiple quantiles test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Fused Adaptive Sampling + Attention Model")
    print("=" * 60)
    
    tests = [
        test_adaptive_quantile_sampler,
        test_adaptive_attention_module,
        test_adaptive_attention_block,
        test_fused_model,
        test_model_with_different_quantiles
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fused model is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
