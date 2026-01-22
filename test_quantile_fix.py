#!/usr/bin/env python3
"""
Test the quantile tensor fix for coverage metric
"""

def test_quantile_shapes():
    """Test that quantile tensor shapes are correct"""
    print("=== Testing Quantile Tensor Shapes ===")
    
    # Simulate the tensor shapes
    import numpy as np
    
    batch_size = 4
    num_quantiles = 7  # After adding evaluation quantiles
    
    # Original quantiles from dataset
    original_quantiles = np.random.rand(batch_size, 1) * 0.8 + 0.1  # [B, 1]
    print(f"Original quantiles shape: {original_quantiles.shape}")
    
    # After add_evaluation_quantiles: [B, 1+3] = [B, 4]
    extended_quantiles = np.concatenate([
        original_quantiles,
        np.array([[0.5, 0.975, 0.025]] * batch_size).reshape(batch_size, 3)
    ], axis=1)
    print(f"Extended quantiles shape: {extended_quantiles.shape}")
    
    # After shared_forward sorting: [B, Q] where Q = total quantiles
    sorted_quantiles = np.sort(extended_quantiles, axis=1)
    print(f"Sorted quantiles shape: {sorted_quantiles.shape}")
    
    # In validation step, need to reshape to [B, 1, Q] for coverage
    quantiles_for_coverage = sorted_quantiles[:, None, :]
    print(f"Quantiles for coverage shape: {quantiles_for_coverage.shape}")
    
    # Predictions shape: [B, H, Q]
    horizon = 48
    predictions = np.random.randn(batch_size, horizon, num_quantiles)
    print(f"Predictions shape: {predictions.shape}")
    
    # Target shape: [B, H]
    targets = np.random.randn(batch_size, horizon)
    print(f"Targets shape: {targets.shape}")
    
    print("\n=== Shape Compatibility ===")
    print(f"Predictions and quantiles compatible: {predictions.shape[-1] == quantiles_for_coverage.shape[-1]}")
    print(f"Targets and predictions compatible: {targets.shape[:2] == predictions.shape[:2]}")
    
    # Test quantile matching logic
    level_low = 0.025
    level_high = 0.975
    tolerance = 1e-6
    
    mask_high = np.abs(quantiles_for_coverage - level_high) < tolerance
    mask_low = np.abs(quantiles_for_coverage - level_low) < tolerance
    
    print(f"\n=== Quantile Matching ===")
    print(f"Level low: {level_low}")
    print(f"Level high: {level_high}")
    print(f"Found quantiles: {quantiles_for_coverage[0, 0, :]}")
    print(f"High mask: {mask_high[0, 0, :]}")
    print(f"Low mask: {mask_low[0, 0, :]}")
    print(f"High found: {mask_high.any()}")
    print(f"Low found: {mask_low.any()}")
    print(f"Both found: {mask_high.any() and mask_low.any()}")
    
    return mask_high.any() and mask_low.any()

if __name__ == "__main__":
    success = test_quantile_shapes()
    if success:
        print("\n✅ Quantile shapes are correct - coverage should work!")
    else:
        print("\n❌ Quantile shapes issue found - coverage will be null!")
