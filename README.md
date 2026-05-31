# cuGPT

this will be a long journey and it will be about creating a small GPT like LLM that can run on a single GPU. Tag along as I learn over how to implement this without creating a mess.

## The Current Timeline

I am currently gathering and reading through sources over how I can implement this. So far I have decided to follow the simple rule: "C manages, CUDA computes". The main idea from what I've understood is to make C manage CUDA calls in the order of a transformer and do proper memory allocations before the transformer run, and the CUDA kernels are responsible for doing the inference calculations. I hope to protect this oversimplification and keep this as readable and maintainable as possible.