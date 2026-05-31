#!/bin/bash
# Usage: ./scripts/test_build.sh <layer_name> (e.g., ./scripts/test_build.sh mlp)

LAYER=$1

if [ -z "$LAYER" ]; then
    echo "❌ Error: Please specify a layer name to test."
    echo "Usage: ./scripts/test_build.sh <layer_name>. e.g., ./scripts/test_build.sh mlp"
    exit 1
fi

echo "🔨 Building Test Suite for: $LAYER"

# We compile:
# 1. The specific test file (tests/test_layernorm.cu)
# 2. Every layer implementation (src/layers/*.cu) so the linker can find the logic
# 3. Every common utility (common/src/*.cu)
nvcc -I./common/include \
     -I./src/layers \
     tests/test_layer.cu \
     tests/test_${LAYER}.cu \
     src/layers/${LAYER}.cu \
     common/src/*.cu \
     -o bin/test_${LAYER}

if [ $? -eq 0 ]; then
    echo "✅ Build Successful! Run: ./bin/test_${LAYER}"
else
    echo "❌ Build Failed"
    exit 1
fi