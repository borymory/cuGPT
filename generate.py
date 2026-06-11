import subprocess
import tiktoken

def generate_text(prompt, checkpoint_path="checkpoints/gpt2_124m.bin"):
    # 1. Encode prompt to token IDs using tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)
    
    # Convert token integers to string arguments for C++
    token_args = [str(t) for t in prompt_tokens]
    
    # 2. Build the command: ./build/gpt2_inference <checkpoint> <tokens...>
    cmd = ["./bin/gpt2_inference", checkpoint_path] + token_args
    
    # 3. Launch the C++ executable and capture its output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("C++ Engine Error:", result.stderr)
        return
    
    # 4. Parse the output token IDs printed by C++
    output_tokens_str = result.stdout.strip().split()
    output_tokens = [int(t) for t in output_tokens_str]
    
    # 5. Decode back to human text and print
    generated_text = enc.decode(output_tokens)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    generate_text("The sky is")