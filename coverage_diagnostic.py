#!/usr/bin/env python3
"""
Standalone diagnostic to identify why coverage is null
Analyzes the Coverage metric logic without requiring external dependencies
"""

def analyze_coverage_logic():
    """Analyze the Coverage metric implementation logic"""
    print("=== Coverage Metric Analysis ===")
    
    # Simulate the Coverage metric parameters
    level = 0.95
    level_low = (1.0 - level) / 2  # 0.025
    level_high = 1.0 - level_low   # 0.975
    
    print(f"Coverage level: {level}")
    print(f"Expected low quantile: {level_low}")
    print(f"Expected high quantile: {level_high}")
    
    # Analysis of potential issues
    print("\n=== Potential Issues ===")
    
    print("\n1. Denominator Issue:")
    print("   - If denominator == 0, compute() returns nan")
    print("   - Denominator only increments when update() is called successfully")
    print("   - If update() always returns early, denominator stays 0")
    
    print("\n2. Update() Early Returns:")
    print("   - NaN/Inf in predictions or targets")
    print("   - Missing required quantiles (level_low, level_high)")
    print("   - Quantile matching failure")
    
    print("\n3. Quantile Matching Issue:")
    print("   - Uses torch.isclose() with atol=1e-6")
    print("   - If actual quantiles don't match expected values closely, masks will be empty")
    print("   - mask_high.any() and mask_low.any() must both be True")
    
    print("\n4. Data Flow Issues:")
    print("   - add_evaluation_quantiles() must be called before forward pass")
    print("   - Quantiles must be sorted in shared_forward()")
    print("   - Predictions must align with quantile dimensions")

def simulate_coverage_scenarios():
    """Simulate different scenarios that could cause null coverage"""
    print("\n=== Scenario Simulation ===")
    
    # Scenario 1: No updates at all
    print("\n--- Scenario 1: No Updates ---")
    numerator = 0.0
    denominator = 0.0
    result = float('nan') if denominator <= 0 else numerator / denominator
    print(f"Result with no updates: {result}")
    print(f"Is null: {result != result}")
    
    # Scenario 2: NaN inputs cause early return
    print("\n--- Scenario 2: NaN Inputs ---")
    has_nan = True
    if has_nan:
        print("Update() returns early, denominator stays 0")
        result = float('nan')
    print(f"Result with NaN inputs: {result}")
    
    # Scenario 3: Missing quantiles
    print("\n--- Scenario 3: Missing Quantiles ---")
    expected_low = 0.025
    expected_high = 0.975
    actual_quantiles = [0.1, 0.5, 0.9]  # Missing required quantiles
    
    has_low = any(abs(q - expected_low) < 1e-6 for q in actual_quantiles)
    has_high = any(abs(q - expected_high) < 1e-6 for q in actual_quantiles)
    
    print(f"Has low quantile ({expected_low}): {has_low}")
    print(f"Has high quantile ({expected_high}): {has_high}")
    print(f"Update() skipped: {not (has_low and has_high)}")
    
    if not (has_low and has_high):
        result = float('nan')
    print(f"Result with missing quantiles: {result}")

def analyze_model_integration():
    """Analyze how Coverage integrates with the model"""
    print("\n=== Model Integration Analysis ===")
    
    print("\n1. Initialization:")
    print("   - Coverage(level=0.95) created in AnyQuantileForecaster.__init__()")
    print("   - level_low=0.025, level_high=0.975")
    
    print("\n2. Validation Step:")
    print("   - batch['quantiles'] = self.val_coverage.add_evaluation_quantiles(batch['quantiles'])")
    print("   - Adds [0.5, 0.975, 0.025] to existing quantiles")
    print("   - shared_forward() sorts quantiles")
    
    print("\n3. Update Call:")
    print("   - self.val_coverage(preds_full, target_full, q=quantiles)")
    print("   - preds_full: [B, H, Q] tensor")
    print("   - target_full: [B, H] tensor")
    print("   - q: [B, 1, Q] tensor")
    
    print("\n4. Potential Failure Points:")
    print("   - Quantile tensor shape mismatch")
    print("   - Sorting changes quantile order")
    print("   - Device/CPU tensor mismatches")
    print("   - Floating point precision issues")

def check_quantile_precision():
    """Check for floating point precision issues"""
    print("\n=== Quantile Precision Analysis ===")
    
    level = 0.95
    level_low = (1.0 - level) / 2
    level_high = 1.0 - level_low
    
    print(f"Level low calculation: (1.0 - {level}) / 2 = {level_low}")
    print(f"Level high calculation: 1.0 - {level_low} = {level_high}")
    
    # Check floating point representation
    print(f"\nFloating point representation:")
    print(f"level_low as string: {str(level_low)}")
    print(f"level_high as string: {str(level_high)}")
    
    # Simulate torch.isclose behavior
    tolerance = 1e-6
    test_quantiles = [0.025, 0.5, 0.975]
    
    print(f"\nQuantile matching with tolerance {tolerance}:")
    for q in test_quantiles:
        matches_low = abs(q - level_low) < tolerance
        matches_high = abs(q - level_high) < tolerance
        print(f"  {q}: matches_low={matches_low}, matches_high={matches_high}")

def main():
    print("Coverage Null Issue Diagnostic")
    print("=" * 50)
    
    analyze_coverage_logic()
    simulate_coverage_scenarios()
    analyze_model_integration()
    check_quantile_precision()
    
    print("\n=== Most Likely Causes ===")
    print("1. Denominator never incremented (update() always returns early)")
    print("2. Quantile matching failure due to precision or ordering issues")
    print("3. NaN/Inf in predictions or targets")
    print("4. Missing required quantiles after sorting/concatenation")
    
    print("\n=== Recommended Debug Steps ===")
    print("1. Add debug prints in Coverage.update() to check early returns")
    print("2. Log quantile values before and after add_evaluation_quantiles()")
    print("3. Check for NaN/Inf in predictions and targets")
    print("4. Verify quantile tensor shapes and devices")
    print("5. Test with a minimal batch to isolate the issue")

if __name__ == "__main__":
    main()
