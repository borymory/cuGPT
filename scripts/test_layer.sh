#!/bin/bash
# Usage: ./scripts/test_layer.sh <layer_name> (e.g., ./scripts/test_build.sh mlp)

LAYER=$1

if [ -z "$LAYER" ]; then
    echo "❌ Error: Please specify a layer name to test."
    echo "Usage: ./scripts/test_build.sh <layer_name>. e.g., ./scripts/test_build.sh mlp"
    exit 1
fi

echo "🔨 Building Test Suite for: $LAYER"

# Compile:
# common headers and .cu files
# test_(layername).cu file
# (layername).cu file in src/layers
nvcc -I./common/include \
     -I./src/layers \
     tests/test_${LAYER}.cu \
     src/layers/${LAYER}.cu \
     common/src/*.cu \
     -lcublas \
     -o bin/test_${LAYER}

if [ $? -eq 0 ]; then
    echo "✅ Build Successful! Run: ./bin/test_${LAYER}"
else
    echo "❌ Build Failed"
    exit 1
fi