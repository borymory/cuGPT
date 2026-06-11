#!/bin/bash
# Usage: ./scripts/build_inference.sh

echo "🔨 Building Main Inference Loop"

# Compile:
# common headers and .cu files
nvcc -I./common/include \
     common/src/*.cu \
     gpt2_inference.cu
     -lcublas \
     -o bin/gpt2_inference

if [ $? -eq 0 ]; then
    echo "✅ Build Successful! Run: ./bin/gpt2_inference"
else
    echo "❌ Build Failed"
    exit 1
fi