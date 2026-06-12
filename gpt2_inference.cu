#include "hpc_utils.cuh"
#include <stdio.h>
#include <stdlib.h>

// Reference
// H: number of heads
// C = d_model, the whole embedding dimension
// d_head: =C/H
// B: batch
// N or T: sequence_length
// max_T, max_seq_len, the maximum sequence length that can be processed
// L: number of layers (number of forward passes)

//
// KERNELS
//
__global__ void embedding_v1(const int *inputs, float *out, const float *wte, const float *wpe, const int B, const int T, const int C, const int max_length) {
    // View inputs and out as 1D
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * C;

    // grid-stride loop
    for (int idx = tid; idx < total_elements; idx += blockDim.x * gridDim.x) {
        
        int c_idx = idx % C;        // which channel in the token
        int token_idx = idx / C;    // which token in the sequence
        int token_id = inputs[token_idx];
        int pos_idx = token_idx % T;

        out[idx] = wte[token_id * C + c_idx] + wpe[pos_idx * C + c_idx];
    }
}

// Kernel Launchers
void launch_embedding_v1(int *inputs, float *out, float *wte, float *wpe, int B, int T, int C, int max_length, cudaStream_t stream) {
    int total_elements = B * T * C;

    int thread_count = 256;
    int block_count = CEIL_DIV(total_elements, thread_count);

    embedding_v1<<<block_count, thread_count, 0, stream>>>(inputs, out, wte, wpe, B, T, C, max_length);
    CHECK_LAST_CUDA_ERROR();
}

//
// MODEL DEFINITION
//

// for comparing weight config with model config
typedef struct {
    int32_t magic;
    int32_t max_seq_len;
    int32_t vocab_size;
    int32_t layers;
    int32_t heads;
    int32_t channels;
} file_header;

typedef enum {
    EMBED_OUT_IDX = 0,

    NUM_ACTIVATIONS
} ActivationIndex;

typedef enum {
    WTE_IDX = 0,
    WPE_IDX,
    LN_1_ALPHA_IDX,
    LN_1_BETA_IDX,

    PROJ_W_Q_IDX,
    PROJ_W_K_IDX,
    PROJ_W_V_IDX,
    PROJ_B_Q_IDX,
    PROJ_B_K_IDX,
    PROJ_B_V_IDX,
    PROJ_W_O_IDX,
    PROJ_B_O_IDX,

    LN_2_ALPHA_IDX,
    LN_2_BETA_IDX,

    FFN_W1_IDX,
    FFN_B1_IDX,
    FFN_W2_IDX,
    FFN_B2_IDX,

    LN_FINAL_ALPHA_IDX,
    LN_FINAL_BETA_IDX,

    NUM_TENSORS
} TensorIndex;

typedef struct {
    int max_seq_len;// max sequence length (e.g., 1024)
    int batch;      // max batch count
    int vocab_size; // vocab_size (e.g., 50257)
    int layers;     // num of layers (e.g., 12)
    int heads;      // num of heads (e.g., 12)
    int channels;   // channel count (e.g., 768)
} model_config;

// holds model's weight ptrs
typedef struct {
    float* wte; // [vocab_size, C]
    float* wpe; // [max_T, C]

    // ---- For layers 0 to 11 -----
    float* ln_1_alpha;  // [L, C]
    float* ln_1_beta;   // [L, C]

    float* proj_w_q;    // [L, C, H, d_head]
    float* proj_w_k;    // [L, C, H, d_head]
    float* proj_w_v;    // [L, C, H, d_head]
    float* proj_b_q;    // [L, C]
    float* proj_b_k;    // [L, C]
    float* proj_b_v;    // [L, C]

    float* proj_w_o;    // [L, C, C]
    float* proj_b_o;    // [L, C]

    float* ln_2_alpha;  // [L, C]
    float* ln_2_beta;   // [L, C]

    float* ffn_w1;  // [L, C, 4C]
    float* ffn_b1;  // [L, 4C]
    float* ffn_w2;  // [L, 4C, C]
    float* ffn_b2;  // [L, C]
    // ------------------------

    float* ln_final_alpha;  // [C]
    float* ln_final_beta;   // [C]
} model_parameters;

typedef struct {
    float* embedding_out; // [B, T, C]
} model_activations;

typedef struct {
    float* key_cache;   // [L, max_B, H, max_T, d_head]
    float* value_cache; // [L, max_B, H, max_T, d_head]
} KV_cache;

typedef struct {
    model_config config;
    model_activations d_activations;    // sliced device activations
    model_parameters d_weights; // sliced device weights
    KV_cache d_kv_cache;        // slices device KV cache

    float* h_weights_base;      // base ptr to host side weights
    float* d_weights_base;      // base ptr to device side weights
    float* d_activations_base;  // base ptr to device side activations
    float* d_kv_cache_base;     // base ptr to device side KV cache

    // tokenized inputs
    int* h_prompt;
    int current_seq_len;
    int* d_prompt;

    void prompt(char* argv[], const int prompt_len) {
        current_seq_len = prompt_len;
        size_t max_context_size = (size_t)config.max_seq_len * sizeof(int);
        h_prompt = (int*)malloc(max_context_size);
        if (!h_prompt) {
            fprintf(stderr, "Failed to allocate CPU memory for prompt tokens.\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < prompt_len; ++i) {
            h_prompt[i] = atoi(argv[i+2]);
        }

        CUDA_CHECK(cudaMalloc((void**)&d_prompt, max_context_size));
        CUDA_CHECK(cudaMemcpy(d_prompt, h_prompt, prompt_len * sizeof(int), cudaMemcpyHostToDevice));

        fprintf(stderr, "Model Prompted. Prompt ptr located at: %p\n", (void*)d_prompt);
    }
} model;

void calculate_param_buffer_size (size_t* param_size, model_config config) {
    int max_T = config.max_seq_len;
    int V = config.vocab_size;
    int L = config.layers;
    int C = config.channels;
    
    // EMBEDDING
    param_size[WTE_IDX] = (size_t)V * C; // token embedding
    param_size[WPE_IDX] = (size_t)max_T * C; // positional embedding

    // LAYERNORM w1, b1
    param_size[LN_1_ALPHA_IDX] = (size_t)L * C; 
    param_size[LN_1_BETA_IDX] = (size_t)L * C;

    // ATTENTION proj_w_kqv, proj_b_kqv, proj_w_o, proj_b_o
    for (int tensor_idx = 4; tensor_idx < 7; ++tensor_idx) {
        param_size[tensor_idx] = (size_t)L * C * C;
    }

    for (int tensor_idx = 7; tensor_idx < 10; ++tensor_idx) {
        param_size[tensor_idx] = (size_t)L * C;
    }

    // ATTENTION proj_w_o, proj_b_o
    param_size[PROJ_W_O_IDX] = (size_t)L * C * C;
    param_size[PROJ_B_O_IDX] = (size_t)L * C;

    // LAYERNORM w2, b2
    param_size[LN_2_ALPHA_IDX] = (size_t)L * C; 
    param_size[LN_2_BETA_IDX] = (size_t)L * C;

    // FFN w1, b1, w2, b2
    param_size[FFN_W1_IDX] = (size_t)L * C * 4 * C;
    param_size[FFN_B1_IDX] = (size_t)L * 4 * C;
    param_size[FFN_W2_IDX] = (size_t)L * 4 * C * C;
    param_size[FFN_B2_IDX] = (size_t)L * C;

    // LAYERNORM fw, fb
    param_size[LN_FINAL_ALPHA_IDX] = (size_t)C;
    param_size[LN_FINAL_BETA_IDX] = (size_t)C;
}

void calculate_activ_buffer_size (size_t* activ_size, model_config config) {
    int max_T = config.max_seq_len;
    int V = config.vocab_size;
    int L = config.layers; // overwrite at each layer (seems possible)
    int C = config.channels;

    activ_size[EMBED_OUT_IDX] = (size_t)max_T * C;
    // populate activ_size array with further hidden activations
}

// Implementation inspired from Andrej Karpathy's llm.c repo :)
// Returns a base ptr to the GPU buffer and slices the GPU buffers to tensor parameters using model_parameters or model_activations
float* malloc_and_point_to_weights(size_t* param_size, model_parameters* params) {

    size_t total_param_size = 0;
    for (size_t i = 0; i < NUM_TENSORS; ++i) {
        total_param_size += param_size[i];
    }

    float* params_memory;
    CUDA_CHECK(cudaMalloc((void**)&params_memory, total_param_size * sizeof(float)));

    // array of adresses to model weight pointers
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln_1_alpha, &params->ln_1_beta, &params->proj_w_q, &params->proj_w_k, &params->proj_w_v, &params->proj_b_q,
        &params->proj_b_k, &params->proj_b_v, &params->proj_w_o, &params->proj_b_o, &params->ln_2_alpha, &params->ln_2_beta, &params->ffn_w1, &params->ffn_b1,
        &params->ffn_w2, &params->ffn_b2, &params->ln_final_alpha, &params->ln_final_beta
    };
    int num_ptrs = sizeof(ptrs) / sizeof(ptrs[0]);
    if (num_ptrs != NUM_TENSORS) {
        fprintf(stderr, "Error: Mismatch in parameter tensor and pointer count! Tensor: %d, Ptr: %d\n", NUM_TENSORS, num_ptrs);
        cudaFree(&params_memory);
        exit(EXIT_FAILURE);
    }

    float* base_offset = params_memory;
    for (size_t i = 0; i < NUM_TENSORS; ++i) {
        *(ptrs[i]) = base_offset;
        base_offset += param_size[i];
    }

    return params_memory;
}

float* malloc_and_point_to_activations(size_t* activ_size, model_activations* activations) {

    size_t total_activ_size = 0;
    for (size_t i = 0; i < NUM_ACTIVATIONS; ++i) {
        total_activ_size += activ_size[i];
    }

    float* activs_memory;
    CUDA_CHECK(cudaMalloc((void**)&activs_memory, total_activ_size * sizeof(float)));

    // array of adresses to model activ pointers
    float** ptrs[] = {
        &activations->embedding_out // add more pointer adresses as more activations need to be sliced
    };
    int num_ptrs = sizeof(ptrs) / sizeof(ptrs[0]);
    if (num_ptrs != NUM_ACTIVATIONS) {
        fprintf(stderr, "Error: Mismatch in activation and pointer count! Activ: %d, Ptr: %d\n", NUM_ACTIVATIONS, num_ptrs);
        cudaFree(&activs_memory);
        exit(EXIT_FAILURE);
    }

    float* base_offset = activs_memory;
    for (size_t i = 0; i < NUM_ACTIVATIONS; ++i) {
        *(ptrs[i]) = base_offset;
        base_offset += activ_size[i];
    }

    return activs_memory;
}

// Initializes the model on GPU with weights loaded
model init_model(model_config config, const char* checkpoint_path) {
    model m;
    m.config = config;

    // calculate the total memory needed for specific model
    size_t activ_size[NUM_ACTIVATIONS];
    size_t param_size[NUM_TENSORS];
    calculate_param_buffer_size(param_size, m.config);
    calculate_activ_buffer_size(activ_size, m.config);

    // create GPU side buffer with correct weight/activation ptrs and base ptr
    m.d_weights_base = malloc_and_point_to_weights(param_size, &m.d_weights);
    m.d_activations_base = malloc_and_point_to_activations(activ_size, &m.d_activations);

    // load weights onto CPU, then copy to GPU
    size_t total_elements = 0;
    for (int i = 0; i < NUM_TENSORS; ++i) {
        total_elements += param_size[i];
    }
    size_t total_bytes = total_elements * sizeof(float);

    FILE* file = fopen(checkpoint_path, "rb");
    if (!file) {
        fprintf(stderr, "Error: Failed to open checkpoint file at %s\n", checkpoint_path);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    file_header header;
    if (fread(&header, sizeof(file_header), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read checkpoint header.\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    if (header.magic != 20241027) {
        fprintf(stderr, "Error: Mismatch in magic nubmer!\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    if (header.layers != (config.layers) ||
        header.channels != config.channels ||
        header.vocab_size != config.vocab_size) {
        fprintf(stderr, "Error: Mismatch in model config!\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    size_t expected_size = sizeof(file_header) + total_bytes;
    // Free host side weights
    free(m.h_weights_base);
    if (file_size != expected_size) {
        fprintf(stderr, "Error: Mismatch on file and expected size!\n");
        fprintf(stderr, "Expected: %zu bytes, Actual: %zu bytes", expected_size, file_size);
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    fseek(file, sizeof(file_header), SEEK_SET);
    m.h_weights_base = (float*)malloc(total_bytes);
    if (!m.h_weights_base) {
        fprintf(stderr, "Error: Failed to allocate CPU memory for weights.\n");
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }

    size_t read_count = fread(m.h_weights_base, sizeof(float), total_elements, file);
    if (read_count != total_elements) {
        fprintf(stderr, "Error: Failed to read %zu elements. Read: %zu elements", total_elements, read_count);
        fclose(file);
        cudaFree(m.d_weights_base);
        exit(EXIT_FAILURE);
    }
    fclose(file);

    CUDA_CHECK(cudaMemcpy(m.d_weights_base, m.h_weights_base, total_bytes, cudaMemcpyHostToDevice));

    free(m.h_weights_base);
    m.h_weights_base = NULL;

    fprintf(stderr, "Model Initialized. GPU weight base: %p\n", (void*)m.d_weights_base);
    fprintf(stderr, "GPU activations base: %p\n", (void*)m.d_activations_base);

    return m;
}

void free_model(model* m) {
    cudaFree(m->d_weights_base);    // free device side weights
    cudaFree(m->d_activations_base);    // free device side activations
    cudaFree(m->d_kv_cache_base);   // free device side KV cache
    cudaFree(m->d_prompt);
    free(m->h_prompt);
}

//
// Forward Pass (B == 1) for now
//
void prefill_forward(model* m, cudaStream_t stream) {
    int current_seq_len = m->current_seq_len;
    int C = m->config.channels;
    int max_seq_len = m->config.max_seq_len;
    int L = m->config.layers;

    // Preprocessing
    int* user_prompt = m->d_prompt;
    float* hidden_embedding = m->d_activations.embedding_out; // of shape [BT, C]
    float* wte = m->d_weights.wte;
    float* wpe = m->d_weights.wpe;
    launch_embedding_v1(user_prompt, hidden_embedding, wte, wpe, 1, current_seq_len, C, max_seq_len, stream);

    // L-1 for proper layer offest indexology
    for (int i = 0; i < L-1; ++i) {
        // residual(x)

        // x = layernorm
        // x = attn

        // x' = x + residual(x)

        // x = layernorm
        // x = mlp

        // x = x + x'
    }

    // x final layernorm
    // x softmax
    // x sampler
}

//
// Main Inference Loop
//

int main(int argc, char* argv[]) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // argc: argument count
    // argv[0] = program name
    // argv[1] = checkpoint path
    // argv[2...] = token IDs

    if (argc < 3) {
        fprintf(stderr, "Error: Missing Arguments.\n");
        fprintf(stderr, "Usage: %s <checkpoint_path> <token 1> <token 2> ...\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Init model
    const char* checkpoint_path = argv[1]; // "checkpoints/gpt2_124m.bin"
    model_config GPT2Config = {1024, 1, 50257, 12, 12, 768}; // batch = 1
    model cuGPT = init_model(GPT2Config, checkpoint_path);

    // Prompt model
    int prompt_len = argc - 2;
    cuGPT.prompt(argv, prompt_len);
    
    // Inference loop:

    // Prefill Phase
    prefill_forward(&cuGPT, stream);

    // Decode Phase
    // include check whether <EOS> token is reached

    // Print last tokens
    for (int i = 0; i < prompt_len; ++i) {
        printf("%d ", cuGPT.h_prompt[i]);
    }

    free_model(&cuGPT);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}