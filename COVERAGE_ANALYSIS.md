# Coverage Null Issue Analysis and Solution

## Problem Summary
The coverage metric is returning `null` (NaN) values during model training/validation.

## Root Cause Analysis

Based on the diagnostic analysis, the most likely causes are:

1. **Denominator never incremented** - The `Coverage.update()` method returns early before updating the numerator/denominator
2. **Missing required quantiles** - The quantile matching logic fails to find the required low (0.025) and high (0.975) quantiles
3. **NaN/Inf in predictions or targets** - Early return due to invalid tensor values
4. **Quantile tensor shape/device mismatches** - The quantile tensor doesn't match expected format

## Key Findings

### Coverage Metric Logic
- `Coverage.compute()` returns `nan` when `denominator == 0`
- `denominator` only increments when `update()` completes successfully
- `update()` returns early if:
  - NaN/Inf detected in inputs
  - Required quantiles (level_low, level_high) not found
  - Quantile matching fails

### Quantile Matching Issue
- Uses `torch.isclose()` with tolerance `1e-6`
- Expected quantiles: `level_low = 0.025`, `level_high = 0.975` (for 95% coverage)
- Floating point precision: `level_low = 0.025000000000000022`
- The metric adds evaluation quantiles: `[0.5, 0.975, 0.025]` to existing quantiles

### Model Integration
- `add_evaluation_quantiles()` called in validation/test steps
- Quantiles get sorted in `shared_forward()`
- Potential issues with quantile ordering after sorting

## Solution Implemented

### 1. Added Debug Logging
Modified `metrics/metrics.py` to add comprehensive debug logging:

```python
# In Coverage.update():
print(f"DEBUG Coverage: level={self.level}, level_low={self.level_low}, level_high={self.level_high}")

# When quantiles are missing:
print(f"DEBUG Coverage: Missing quantiles!")
print(f"  Looking for: low={self.level_low}, high={self.level_high}")
print(f"  Found quantiles: {q_flat}")

# On successful update:
print(f"DEBUG Coverage: Successful update - numerator={self.numerator}, denominator={self.denominator}")

# In Coverage.compute():
print(f"DEBUG Coverage: Computing - numerator={self.numerator}, denominator={self.denominator}")
```

### 2. Diagnostic Scripts Created
- `coverage_diagnostic.py` - Standalone analysis of coverage logic
- `debug_coverage_detailed.py` - Working debug version of coverage metric
- `test_coverage_debug.py` - Test script for model integration

## Next Steps

### To Identify the Exact Issue:
1. Run the model with debug logging enabled:
   ```bash
   docker run --gpus all --rm -it --shm-size=8g -e TORCH_FLOAT32_MATMUL_PRECISION=medium \
   -v "${PWD}:/workspace/any-quantile" -w /workspace/any-quantile any_quantile:latest \
   python run.py --config=config/nbeatsaq-attention-mhlv-fast.yaml
   ```

2. Look for debug output showing:
   - Whether `update()` is being called
   - If quantiles are missing
   - If NaN/Inf is detected
   - Final numerator/denominator values

### Potential Fixes (Based on Debug Output):

**If quantiles are missing:**
- Check `add_evaluation_quantiles()` implementation
- Verify quantile sorting doesn't disrupt required values
- Increase tolerance in `torch.isclose()`

**If NaN/Inf detected:**
- Add better input validation
- Clamp predictions/targets to valid ranges
- Investigate model output issues

**If denominator stays 0:**
- Verify `update()` is actually being called
- Check for early returns due to warnings
- Ensure metric is properly initialized

## Files Modified
- `metrics/metrics.py` - Added debug logging to Coverage class

## Files Created
- `coverage_diagnostic.py` - Logic analysis
- `debug_coverage_detailed.py` - Debug version
- `test_coverage_debug.py` - Integration test

## Expected Debug Output
When run, the debug logging will show exactly why coverage is null:
```
DEBUG Coverage: level=0.95, level_low=0.025, level_high=0.975
DEBUG Coverage: Missing quantiles!
  Looking for: low=0.025, high=0.975
  Found quantiles: [0.1 0.3 0.7 0.9 0.5 0.975 0.025]
DEBUG Coverage: Computing - numerator=0.0, denominator=0.0
DEBUG Coverage: Returning NaN (denominator is 0)
```

This will pinpoint whether the issue is quantile matching, NaN values, or another problem.
