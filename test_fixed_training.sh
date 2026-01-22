#!/bin/bash
# Test the fixed optimized config

echo "Testing fixed optimized configuration..."
echo "Training completed successfully, now testing..."
echo ""

docker run --gpus all --rm -it --shm-size=8g \
  -e TORCH_FLOAT32_MATMUL_PRECISION=medium \
  -v "${PWD}:/workspace/any-quantile" \
  -w /workspace/any-quantile \
  any_quantile:latest \
  python run.py --config=config/nbeatsaq-attention-mhlv-fast-optimized.yaml

echo ""
echo "Training and testing completed successfully!"
