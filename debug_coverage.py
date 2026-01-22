#!/usr/bin/env python3
"""
Diagnostic script to identify why coverage is null
"""

import torch
import warnings
from metrics import Coverage

def test_coverage_basic():
    """Test basic coverage functionality"""
    print("=== Testing Coverage Metric ===")
    
    # Create coverage metric
    coverage = Coverage(level=0.95)
    print(f"Coverage level: {coverage.level}")
    print(f"Level low: {coverage.level_low}")
    print(f"Level high: {coverage.level_high}")
    
    # Test data
    batch_size = 4
    horizon = 48
    quantiles = torch.tensor([0.025, 0.5, 0.975])  # Low, median, high
    
    # Create predictions and targets
    preds = torch.randn(batch_size, horizon, len(quantiles)) * 10 + 50
    target = torch.randn(batch_size, horizon) * 10 + 50
    
    # Add evaluation quantiles
    quantiles_extended = coverage.add_evaluation_quantiles(quantiles)
    print(f"Original quantiles: {quantiles}")
    print(f"Extended quantiles: {quantiles_extended}")
    
    # Create corresponding predictions for extended quantiles
    preds_extended = torch.randn(batch_size, horizon, len(quantiles_extended)) * 10 + 50
    
    print(f"Preds shape: {preds_extended.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Quantiles shape: {quantiles_extended.shape}")
    
    # Test update
    print("\n--- Testing update ---")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coverage.update(preds_extended, target, quantiles_extended)
        if w:
            for warning in w:
                print(f"Warning: {warning.message}")
        else:
            print("No warnings during update")
    
    # Test compute
    print("\n--- Testing compute ---")
    result = coverage.compute()
    print(f"Coverage result: {result}")
    print(f"Numerator: {coverage.numerator}")
    print(f"Denominator: {coverage.denominator}")
    
    return result

def test_coverage_edge_cases():
    """Test edge cases that might cause null coverage"""
    print("\n=== Testing Edge Cases ===")
    
    # Test 1: Empty denominator
    print("\n--- Test 1: No updates ---")
    coverage_empty = Coverage(level=0.95)
    result_empty = coverage_empty.compute()
    print(f"Empty coverage result: {result_empty}")
    
    # Test 2: NaN inputs
    print("\n--- Test 2: NaN inputs ---")
    coverage_nan = Coverage(level=0.95)
    preds_nan = torch.tensor([[1.0, 2.0, float('nan')]])
    target_nan = torch.tensor([1.5])
    quantiles_nan = torch.tensor([0.025, 0.5, 0.975])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coverage_nan.update(preds_nan, target_nan, quantiles_nan)
        if w:
            for warning in w:
                print(f"Warning: {warning.message}")
    
    result_nan = coverage_nan.compute()
    print(f"NaN coverage result: {result_nan}")
    
    # Test 3: Missing quantiles
    print("\n--- Test 3: Missing required quantiles ---")
    coverage_missing = Coverage(level=0.95)
    preds_missing = torch.tensor([[1.0, 2.0, 3.0]])  # Only 3 quantiles
    target_missing = torch.tensor([1.5])
    quantiles_missing = torch.tensor([0.1, 0.3, 0.7])  # Missing 0.025 and 0.975
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coverage_missing.update(preds_missing, target_missing, quantiles_missing)
        if w:
            for warning in w:
                print(f"Warning: {warning.message}")
    
    result_missing = coverage_missing.compute()
    print(f"Missing quantiles coverage result: {result_missing}")

def test_quantile_matching():
    """Test quantile matching logic"""
    print("\n=== Testing Quantile Matching ===")
    
    coverage = Coverage(level=0.95)
    level_high = torch.tensor(coverage.level_high)
    level_low = torch.tensor(coverage.level_low)
    
    print(f"Looking for quantiles: low={level_low}, high={level_high}")
    
    # Test exact match
    quantiles_exact = torch.tensor([0.025, 0.5, 0.975])
    mask_high_exact = torch.isclose(quantiles_exact, level_high, atol=1e-6)
    mask_low_exact = torch.isclose(quantiles_exact, level_low, atol=1e-6)
    print(f"Exact match - high mask: {mask_high_exact}, low mask: {mask_low_exact}")
    
    # Test close match
    quantiles_close = torch.tensor([0.024999, 0.5, 0.975001])
    mask_high_close = torch.isclose(quantiles_close, level_high, atol=1e-6)
    mask_low_close = torch.isclose(quantiles_close, level_low, atol=1e-6)
    print(f"Close match - high mask: {mask_high_close}, low mask: {mask_low_close}")
    
    # Test no match
    quantiles_no_match = torch.tensor([0.1, 0.5, 0.9])
    mask_high_no = torch.isclose(quantiles_no_match, level_high, atol=1e-6)
    mask_low_no = torch.isclose(quantiles_no_match, level_low, atol=1e-6)
    print(f"No match - high mask: {mask_high_no}, low mask: {mask_low_no}")

if __name__ == "__main__":
    print("Coverage Diagnostic Script")
    print("=" * 50)
    
    try:
        # Run basic test
        result = test_coverage_basic()
        
        # Run edge case tests
        test_coverage_edge_cases()
        
        # Test quantile matching
        test_quantile_matching()
        
        print(f"\n=== Summary ===")
        print(f"Basic coverage result: {result}")
        print(f"Is null? {torch.isnan(result) if torch.is_tensor(result) else result != result}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
