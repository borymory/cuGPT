import torch
import numpy as np
from transformers import GPT2LMHeadModel

def export_gpt2_large_weights_tensor_grouped(filepath="checkpoints/gpt2_774m.bin"):
    print("Loading GPT-2 Large (774M / ~3GB)...")
    model = GPT2LMHeadModel.from_pretrained("gpt2-large")
    state_dict = model.state_dict()
    
    # Configuration for GPT-2 Large (774M parameters)
    config = {
        "magic": 20241027,
        "max_seq_len": 1024,
        "vocab_size": 50257,
        "layers": 36,       # 124M had 12
        "heads": 20,        # 124M had 12
        "channels": 1280    # 124M had 768
    }

    # Helper to convert tensor to flat float32 array
    def get_numpy(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)

    print(f"Exporting weights to {filepath}...")
    with open(filepath, "wb") as f:
        # 1. Write the 6-integer Header
        header = np.array([
            config["magic"], config["max_seq_len"], config["vocab_size"],
            config["layers"], config["heads"], config["channels"]
        ], dtype=np.int32)
        header.tofile(f)

        # 2. Write Embeddings
        get_numpy(state_dict["transformer.wte.weight"]).tofile(f)
        get_numpy(state_dict["transformer.wpe.weight"]).tofile(f)

        # 3. Write LN1 scale and bias for ALL layers
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.ln_1.weight"]).tofile(f)
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.ln_1.bias"]).tofile(f)

        # 4. Extract, slice, and write Q, K, V weights for ALL layers
        for i in range(config["layers"]):
            qkv_w = state_dict[f"transformer.h.{i}.attn.c_attn.weight"]
            w_q, _, _ = torch.chunk(qkv_w, 3, dim=-1)
            get_numpy(w_q).tofile(f)
            
        for i in range(config["layers"]):
            qkv_w = state_dict[f"transformer.h.{i}.attn.c_attn.weight"]
            _, w_k, _ = torch.chunk(qkv_w, 3, dim=-1)
            get_numpy(w_k).tofile(f)
            
        for i in range(config["layers"]):
            qkv_w = state_dict[f"transformer.h.{i}.attn.c_attn.weight"]
            _, _, w_v = torch.chunk(qkv_w, 3, dim=-1)
            get_numpy(w_v).tofile(f)

        # 5. Extract, slice, and write Q, K, V biases for ALL layers
        for i in range(config["layers"]):
            qkv_b = state_dict[f"transformer.h.{i}.attn.c_attn.bias"]
            b_q, _, _ = torch.chunk(qkv_b, 3, dim=-1)
            get_numpy(b_q).tofile(f)
            
        for i in range(config["layers"]):
            qkv_b = state_dict[f"transformer.h.{i}.attn.c_attn.bias"]
            _, b_k, _ = torch.chunk(qkv_b, 3, dim=-1)
            get_numpy(b_k).tofile(f)
            
        for i in range(config["layers"]):
            qkv_b = state_dict[f"transformer.h.{i}.attn.c_attn.bias"]
            _, _, b_v = torch.chunk(qkv_b, 3, dim=-1)
            get_numpy(b_v).tofile(f)

        # 6. Write Attn Out-Projection weights and biases for ALL layers
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.attn.c_proj.weight"]).tofile(f)
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.attn.c_proj.bias"]).tofile(f)

        # 7. Write LN2 scale and bias for ALL layers
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.ln_2.weight"]).tofile(f)
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.ln_2.bias"]).tofile(f)

        # 8. Write FFN W1 and B1 (MLP Gate/Up) for ALL layers
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.mlp.c_fc.weight"]).tofile(f)
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.mlp.c_fc.bias"]).tofile(f)

        # 9. Write FFN W2 and B2 (MLP Down) for ALL layers
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.mlp.c_proj.weight"]).tofile(f)
        for i in range(config["layers"]):
            get_numpy(state_dict[f"transformer.h.{i}.mlp.c_proj.bias"]).tofile(f)

        # 10. Write Final LN scale and bias
        get_numpy(state_dict["transformer.ln_f.weight"]).tofile(f)
        get_numpy(state_dict["transformer.ln_f.bias"]).tofile(f)

    print(f"Successfully exported weights in Tensor-Grouped layout to {filepath}")

if __name__ == "__main__":
    export_gpt2_large_weights_tensor_grouped("checkpoints/gpt2_774m.bin")