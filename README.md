<p align="center">
  <img src="images/cuGPT_banner.png" width="100%" alt="cuGPT banner">
</p>

# **cuGPT**

Inference engine on GPT-2 (124M) model, in raw C/CUDA. Currently being tested for NVIDIA Tesla T4.

# **Project Goal**

Build GPT-2 inference avoiding existing frameworks.

Constraints:
* Raw C and CUDA kernels
* First readibility and maintainability, second optimization
* Building from bottom to top: Kernels and Functions -> Layer -> Transformer
* Current approach set to first independently validate kernels, then combine.

The goal is understanding design considerations rather than reproducing existing engines.

# **Current Status**

## **Implemented**

- [x] **GPU** Layernorm 
- [x] **GPU** FeedForward Neural Network
- [x] **GPU** Token Embedding 
- [x] **GPU** Positional Embedding
- [x] **GPU** Language Model Head
- [x] **Main Inference Loop** Tensor Loader and Pointer Slicer
- [x] **Main Inference Loop** Model Struct

## **In Progress**

- [ ] **GPU** Online Softmax (with TOP-K)
- [ ] **GPU** Logits Sampler
- [ ] **GPU** Flash Attention + Causal Mask
- [ ] **Main Inference Loop** Forward Pass

# **Quick Start**

**Current development is concentrated at the layer level. E.g., to test a single layer:**

```
git clone https://github.com/borymory/cuGPT.git
```

```
cd cuGPT

chmod +x scripts/test_layer.sh

./scripts/test_layer.sh layernorm
```

# **Roadmap**

- [ ] Inference Loop
  - [x] Weight Loader
  - [x] Model Initializer
  - [ ] Forward Pass Kernels
  - [ ] KV Cache Implementation

Future Direction:

* Training
* Modular tensor parameters and transformer model using struct logic
* Python wrapper for readibility

## **Notes**

Development notes and research live in /research.