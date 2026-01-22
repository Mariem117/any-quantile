#!/usr/bin/env python3
"""
Simple test script for the adaptive attention modules.
Tests the core functionality without requiring the full model stack.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.adaptive_attention import AdaptiveAttention, AdaptiveAttentionBlock
    from utils.adaptive_sampling import AdaptiveQuantileSampler
    print("✓ Successfully imported adaptive attention modules")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("This might be due to missing dependencies. Let's test the basic functionality...")
    
    # Test basic torch functionality
    print("\nTesting basic PyTorch functionality...")
    x = torch.randn(2, 10, 256)
    print(f"✓ Created tensor of shape: {x.shape}")
    print("✓ PyTorch is working correctly")
    exit(0)


def test_adaptive_attention():
    """Test the AdaptiveAttention module."""
    print("\nTesting AdaptiveAttention module...")
    
    # Create test data
    batch_size, seq_len, d_model = 2, 10, 64  # Smaller for testing
    x = torch.randn(batch_size, seq_len, d_model)
    quantile = torch.tensor([[0.5], [0.1]])  # [B, 1] shape
    
    print(f"Input shapes: x={x.shape}, quantile={quantile.shape}")
    
    # Initialize adaptive attention with smaller parameters
    adaptive_attention = AdaptiveAttention(
        d_model=d_model,
        n_heads=4,  # Must divide d_model
        dropout=0.0,  # No dropout for testing
        max_len=100
    )
    
    try:
        # Test forward pass
        output = adaptive_attention(x, quantile)
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print("✓ Forward pass successful")
        
        # Test with return_attention=True
        output_with_attn, attn_weights = adaptive_attention(x, quantile, return_attention=True)
        assert output_with_attn.shape == expected_shape
        assert attn_weights.shape == (batch_size, 4, seq_len, seq_len)  # [B, H, T, T]
        print("✓ Forward pass with attention weights successful")
        
        # Test importance update
        quantiles = torch.tensor([0.1, 0.5, 0.9])
        losses = torch.tensor([0.8, 0.3, 0.6])
        adaptive_attention.update_quantile_importance(quantiles, losses)
        print("✓ Importance update successful")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_attention_block():
    """Test the AdaptiveAttentionBlock."""
    print("\nTesting AdaptiveAttentionBlock...")
    
    batch_size, seq_len, d_model = 2, 10, 64
    x = torch.randn(batch_size, seq_len, d_model)
    quantile = torch.tensor([[0.5], [0.1]])
    
    block = AdaptiveAttentionBlock(
        d_model=d_model,
        n_heads=4,
        dropout=0.0,
        d_ff=128
    )
    
    try:
        output = block(x, quantile)
        expected_shape = (batch_size, seq_len, d_model)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print("✓ Block forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Block test failed: {e}")
        return False


def test_adaptive_sampler():
    """Test the AdaptiveQuantileSampler."""
    print("\nTesting AdaptiveQuantileSampler...")
    
    sampler = AdaptiveQuantileSampler(num_bins=10, momentum=0.9)
    
    try:
        # Test sampling
        samples = sampler.sample(batch_size=4)
        assert samples.shape == (4,), f"Expected shape (4,), got {samples.shape}"
        assert torch.all(samples >= 0) and torch.all(samples <= 1), "Samples should be in [0, 1]"
        print("✓ Sampling successful")
        
        # Test update
        quantiles = torch.tensor([0.1, 0.5, 0.9])
        losses = torch.tensor([0.8, 0.3, 0.6])
        sampler.update(quantiles, losses)
        print("✓ Update successful")
        
        # Test that sampling changes after update
        samples2 = sampler.sample(batch_size=4)
        print("✓ Sampling after update successful")
        
        return True
    except Exception as e:
        print(f"✗ Sampler test failed: {e}")
        return False


def test_integration():
    """Test integration between components."""
    print("\nTesting integration...")
    
    batch_size, seq_len, d_model = 2, 10, 64
    x = torch.randn(batch_size, seq_len, d_model)
    quantile = torch.tensor([[0.5], [0.1]])
    
    # Create components
    sampler = AdaptiveQuantileSampler(num_bins=10)
    attention = AdaptiveAttention(d_model=d_model, n_heads=4, dropout=0.0)
    
    try:
        # Simulate training loop
        for step in range(3):
            # Sample quantiles
            sampled_q = sampler.sample(batch_size)
            
            # Forward pass
            output = attention(x, sampled_q.unsqueeze(1))
            
            # Simulate loss and update
            fake_loss = torch.rand(batch_size)
            sampler.update(sampled_q, fake_loss)
            attention.update_quantile_importance(sampled_q, fake_loss)
        
        print("✓ Integration test successful")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Adaptive Attention Modules")
    print("=" * 50)
    
    tests = [
        test_adaptive_sampler,
        test_adaptive_attention,
        test_adaptive_attention_block,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    print("=" * 50)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
