#!/usr/bin/env python3
"""
Minimal test to reproduce coverage null issue locally
"""

def test_coverage_locally():
    """Test coverage metric with minimal setup"""
    print("=== Local Coverage Test ===")
    
    # Simulate the Coverage metric logic without dependencies
    class MockCoverage:
        def __init__(self, level=0.95):
            self.level = level
            self.level_low = (1.0 - level) / 2
            self.level_high = 1.0 - self.level_low
            self.numerator = 0.0
            self.denominator = 0.0
            
        def add_evaluation_quantiles(self, quantiles):
            """Simulate adding evaluation quantiles"""
            import numpy as np
            quantiles_metric = np.array([0.5, self.level_high, self.level_low])
            quantiles_metric = np.repeat(quantiles_metric[None], repeats=quantiles.shape[0], axis=0)
            combined = np.concatenate([quantiles, quantiles_metric], axis=-1)
            return combined
            
        def update(self, preds, target, q):
            """Simulate update with quantile matching"""
            # Check for required quantiles
            tolerance = 1e-6
            mask_high = np.abs(q - self.level_high) < tolerance
            mask_low = np.abs(q - self.level_low) < tolerance
            
            print(f"Looking for: low={self.level_low}, high={self.level_high}")
            print(f"Found quantiles: {q}")
            print(f"High mask: {mask_high}")
            print(f"Low mask: {mask_low}")
            
            if not (mask_high.any() and mask_low.any()):
                print("MISSING QUANTILES - update skipped")
                return
                
            print("QUANTILES FOUND - updating coverage")
            # Simplified coverage calculation
            self.numerator += 1  # Mock
            self.denominator += target.size
            
        def compute(self):
            print(f"Computing - numerator: {self.numerator}, denominator: {self.denominator}")
            if self.denominator > 0:
                return self.numerator / self.denominator
            else:
                return float('nan')
    
    import numpy as np
    
    # Test scenario 1: Missing quantiles
    print("\n--- Test 1: Missing Quantiles ---")
    coverage1 = MockCoverage(level=0.95)
    
    # Original quantiles that don't include required ones
    original_quantiles = np.array([[0.1, 0.3, 0.7, 0.9]])
    print(f"Original quantiles: {original_quantiles}")
    
    # Add evaluation quantiles
    extended_quantiles = coverage1.add_evaluation_quantiles(original_quantiles)
    print(f"Extended quantiles: {extended_quantiles}")
    
    # Simulate predictions and targets
    preds = np.random.randn(2, 48, 7) * 10 + 50
    target = np.random.randn(2, 48) * 10 + 50
    
    coverage1.update(preds, target, extended_quantiles)
    result1 = coverage1.compute()
    print(f"Result: {result1}")
    
    # Test scenario 2: Correct quantiles
    print("\n--- Test 2: Correct Quantiles ---")
    coverage2 = MockCoverage(level=0.95)
    
    # Start with quantiles that already include required ones
    good_quantiles = np.array([[0.025, 0.5, 0.975]])
    print(f"Good quantiles: {good_quantiles}")
    
    extended_good = coverage2.add_evaluation_quantiles(good_quantiles)
    print(f"Extended good: {extended_good}")
    
    coverage2.update(preds, target, extended_good)
    result2 = coverage2.compute()
    print(f"Result: {result2}")
    
    # Test scenario 3: Floating point precision
    print("\n--- Test 3: Floating Point Precision ---")
    coverage3 = MockCoverage(level=0.95)
    
    # Quantiles with floating point precision issues
    fp_quantiles = np.array([[0.024999999, 0.5, 0.975000001]])
    print(f"FP quantiles: {fp_quantiles}")
    
    extended_fp = coverage3.add_evaluation_quantiles(fp_quantiles)
    print(f"Extended FP: {extended_fp}")
    
    coverage3.update(preds, target, extended_fp)
    result3 = coverage3.compute()
    print(f"Result: {result3}")
    
    print(f"\n=== Summary ===")
    print(f"Test 1 (missing): {result1}")
    print(f"Test 2 (correct): {result2}")
    print(f"Test 3 (precision): {result3}")
    
    return result1, result2, result3

if __name__ == "__main__":
    test_coverage_locally()
