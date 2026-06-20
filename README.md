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

- [x] **GPU** Token Embedding 
- [x] **GPU** Positional Embedding
- [x] **GPU** Layernorm 
- [x] **GPU** Flash Attention + Causal Mask
- [x] **GPU** FeedForward Neural Network
- [x] **GPU** Language Model Head
- [x] **Main Inference Loop** Tensor Loaders and Pointer Slicers
- [x] **Main Inference Loop** Model Struct

## **In Progress**

- [ ] **GPU** Online Softmax (with TOP-K)
- [ ] **GPU** Logits Sampler
- [ ] **Main Inference Loop** Forward Pass

# **Quick Start**

**Current development is now at the inference main loop**


```
git clone https://github.com/borymory/cuGPT.git
```

**To test a single layer:**

```
cd cuGPT

chmod +x scripts/test_layer.sh

./scripts/test_layer.sh layernorm
```

**Currently using tiktoken for tokenization. Build gpt2_inference.cu and then run python to call the executable:**

```
cd cuGPT

mkdir checkpoints

python ./scripts/export_gpt2.py
```

```
chmod +x scripts/build_inference.sh

./scripts/build_inference.sh

python ./generate.py
```


# **Roadmap**

- [ ] Inference Loop
  - [x] Weight Loader
  - [x] Model Initializer
  - [x] KV Cache Implementation
  - [ ] Forward Pass Kernels

Future Direction:

* Training
* Modular tensor parameters and transformer model using struct logic
* Python wrapper for readibility

## **Notes**

Development notes and research live in /research.