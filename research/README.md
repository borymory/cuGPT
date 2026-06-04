## **Work Log**

I chose to store all kernel and organizational related work here. Each markdown file contains personal diagrams/math over that specific layer kernel. This work log serves also as a timeline of cuGPT.

Implemented Language Model head and Sampler. LM Head uses weight tying, reutilizing the embedding weight matrix transposed to obtain logits from the final layernorm. Sampler choice is Online Softmax top-k fused (top k=50 by default), implemented from [this paper, algorithm 4](https://arxiv.org/abs/1805.02867) with some additional changes for removing redundant expf functions and excessive top-k checks calls. The resulting sampler takes in input logits[B, C] and outputs the next tokens[B] for each batch.

The following up task is setting up FlashAttn for the suitable tensor sizes, Q,K,V tensor preparation and casual mask application. Backward kernels are right now out of scope for complexity and accessibility reasons.