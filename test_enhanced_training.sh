#!/bin/bash
# Test enhanced attention training

echo "=== Testing Enhanced Attention Training ==="
echo "Improvements implemented:"
echo "  ✅ Positional encoding for temporal awareness"
echo "  ✅ Enhanced quantile conditioning (Q, K, V)"
echo "  ✅ Fixed quantile broadcasting logic"
echo "  ✅ Robust gradient flow"
echo ""

docker run --gpus all --rm -it --shm-size=8g \
  -e TORCH_FLOAT32_MATMUL_PRECISION=medium \
  -v "${PWD}:/workspace/any-quantile" \
  -w /workspace/any-quantile \
  any_quantile:latest \
  python run.py --config=config/nbeatsaq-attention-enhanced.yaml

echo ""
echo "=== Enhanced Attention Training Complete ==="
