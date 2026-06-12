import subprocess
import tiktoken
import sys

def generate_text(prompt, checkpoint_path="checkpoints/gpt2_124m.bin"):
    # Encode prompt to token IDs using tiktoken
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)
    
    token_args = [str(t) for t in prompt_tokens]
    
    # Build command: ./build/gpt2_inference <checkpoint> <tokens...>
    cmd = ["./bin/gpt2_inference", checkpoint_path] + token_args
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("C++ Engine Error:", result.stderr)
        return
    
    if result.stderr:
        print("---C++ Debug Log---")
        print(result.stderr, file=sys.stderr)
        print("----------------------")
    
    output_tokens_str = result.stdout.strip().split()
    output_tokens = [int(t) for t in output_tokens_str]
    
    generated_text = enc.decode(output_tokens)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    generate_text("The sky is")