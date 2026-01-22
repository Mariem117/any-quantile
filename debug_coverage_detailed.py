#!/usr/bin/env python3
"""
Debug version of Coverage metric to identify why coverage is null
"""

class DebugCoverage:
    """Debug version of Coverage metric with extensive logging"""
    
    def __init__(self, level=0.95):
        self.level = level
        self.level_low = (1.0 - level) / 2
        self.level_high = 1.0 - self.level_low
        self.numerator = 0.0
        self.denominator = 0.0
        self.update_count = 0
        self.skip_count = 0
        self.debug_info = []
        
    def add_evaluation_quantiles(self, quantiles):
        """Debug version of add_evaluation_quantiles"""
        print(f"DEBUG: Original quantiles shape: {quantiles.shape}")
        print(f"DEBUG: Original quantiles: {quantiles.flatten()}")
        
        quantiles_metric = [
            0.5,  # Median
            self.level_high,  # Upper bound
            self.level_low,   # Lower bound
        ]
        
        print(f"DEBUG: Adding metric quantiles: {quantiles_metric}")
        
        # Create tensor for metric quantiles
        import torch
        quantiles_metric = torch.tensor(quantiles_metric)
        quantiles_metric = torch.repeat_interleave(
            quantiles_metric[None], 
            repeats=quantiles.shape[0], 
            dim=0
        )
        quantiles_metric = quantiles_metric.to(quantiles)
        
        combined = torch.cat([quantiles, quantiles_metric], dim=-1)
        print(f"DEBUG: Combined quantiles shape: {combined.shape}")
        print(f"DEBUG: Combined quantiles: {combined.flatten()}")
        
        return combined
    
    def update(self, preds, target, q):
        """Debug version of update with extensive logging"""
        print(f"\n=== Coverage Update #{self.update_count + 1} ===")
        print(f"DEBUG: preds shape: {preds.shape}")
        print(f"DEBUG: target shape: {target.shape}")
        print(f"DEBUG: q shape: {q.shape}")
        print(f"DEBUG: q values: {q.flatten()}")
        
        # Check for NaN/Inf
        import torch
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            print("DEBUG: SKIPPING - NaN/Inf in predictions")
            self.skip_count += 1
            self.debug_info.append("NaN/Inf in predictions")
            return
            
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("DEBUG: SKIPPING - NaN/Inf in targets")
            self.skip_count += 1
            self.debug_info.append("NaN/Inf in targets")
            return
        
        # Align dimensions
        if target.dim() != preds.dim():
            print(f"DEBUG: Aligning dimensions - target dim: {target.dim()}, preds dim: {preds.dim()}")
            target = target[..., None]
            print(f"DEBUG: Aligned target shape: {target.shape}")
        
        # Check quantile matching
        level_high = torch.as_tensor(self.level_high, device=q.device, dtype=q.dtype)
        level_low = torch.as_tensor(self.level_low, device=q.device, dtype=q.dtype)
        
        print(f"DEBUG: Looking for level_high ({level_high}) and level_low ({level_low})")
        
        mask_high = torch.isclose(q, level_high, atol=1e-6)
        mask_low = torch.isclose(q, level_low, atol=1e-6)
        
        print(f"DEBUG: mask_high: {mask_high.flatten()}")
        print(f"DEBUG: mask_low: {mask_low.flatten()}")
        print(f"DEBUG: mask_high.any(): {mask_high.any()}")
        print(f"DEBUG: mask_low.any(): {mask_low.any()}")
        
        if not (mask_high.any() and mask_low.any()):
            print("DEBUG: SKIPPING - Required quantiles missing")
            self.skip_count += 1
            self.debug_info.append(f"Missing quantiles - high_found: {mask_high.any()}, low_found: {mask_low.any()}")
            return
        
        # Calculate coverage
        num_high = mask_high.sum(dim=-1, keepdims=True).clamp(min=1.0)
        num_low = mask_low.sum(dim=-1, keepdims=True).clamp(min=1.0)
        
        preds_high = (preds * mask_high).sum(dim=-1, keepdims=True) / num_high
        preds_low = (preds * mask_low).sum(dim=-1, keepdims=True) / num_low
        
        print(f"DEBUG: preds_high range: [{preds_high.min():.3f}, {preds_high.max():.3f}]")
        print(f"DEBUG: preds_low range: [{preds_low.min():.3f}, {preds_low.max():.3f}]")
        print(f"DEBUG: target range: [{target.min():.3f}, {target.max():.3f}]")
        
        coverage_count = ((target < preds_high) * (target >= preds_low)).sum()
        target_count = torch.numel(target)
        
        print(f"DEBUG: coverage_count: {coverage_count}")
        print(f"DEBUG: target_count: {target_count}")
        print(f"DEBUG: coverage_rate: {coverage_count.float() / target_count}")
        
        self.numerator += coverage_count
        self.denominator += target_count
        self.update_count += 1
        
        print(f"DEBUG: Running total - numerator: {self.numerator}, denominator: {self.denominator}")
    
    def compute(self):
        """Debug version of compute"""
        print(f"\n=== Coverage Compute ===")
        print(f"DEBUG: Total updates: {self.update_count}")
        print(f"DEBUG: Total skips: {self.skip_count}")
        print(f"DEBUG: Skip reasons: {self.debug_info}")
        print(f"DEBUG: Final numerator: {self.numerator}")
        print(f"DEBUG: Final denominator: {self.denominator}")
        
        if self.denominator > 0:
            result = self.numerator / self.denominator
            print(f"DEBUG: Coverage result: {result}")
            return result
        else:
            print("DEBUG: Coverage result: nan (denominator is 0)")
            import torch
            return torch.tensor(float('nan'))

def test_debug_coverage():
    """Test the debug coverage metric"""
    print("Testing Debug Coverage Metric")
    print("=" * 50)
    
    import torch
    
    # Create debug coverage
    coverage = DebugCoverage(level=0.95)
    
    # Test data
    batch_size = 2
    horizon = 4
    quantiles = torch.tensor([[0.1, 0.3, 0.7, 0.9]])  # Missing required quantiles
    
    print(f"Initial quantiles: {quantiles.flatten()}")
    
    # Add evaluation quantiles
    quantiles_extended = coverage.add_evaluation_quantiles(quantiles)
    
    # Create predictions and targets
    preds = torch.randn(batch_size, horizon, quantiles_extended.shape[-1]) * 10 + 50
    target = torch.randn(batch_size, horizon) * 10 + 50
    
    # Test update
    coverage.update(preds, target, quantiles_extended.unsqueeze(1))
    
    # Test compute
    result = coverage.compute()
    
    return result

if __name__ == "__main__":
    test_debug_coverage()
