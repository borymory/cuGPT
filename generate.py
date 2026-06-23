import subprocess
import sys
import bdb
import tiktoken

def generate_text_stream(prompt, checkpoint_path="checkpoints/gpt2_124m.bin"):
    # 1. Encode prompt to token IDs
    enc = tiktoken.get_encoding("gpt2")
    prompt_tokens = enc.encode(prompt)
    token_args = [str(t) for t in prompt_tokens]
    
    # Build command: ./bin/gpt2_inference <checkpoint> <tokens...>
    cmd = ["./bin/gpt2_inference", checkpoint_path] + token_args
    
    # 2. Launch the C++ process in the background with stdout piping
    # We set stderr=None so your C++ debug logs print directly to the screen
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None, 
        text=True,
        bufsize=1 # Line-buffered for fast transfers
    )
    
    print("\n--- Model Generation Starting ---")
    
    # Print the prompt first
    print(prompt, end="", flush=True)

    token_accumulator = ""
    
    # 3. Read the stdout stream character-by-character in real-time
    try:
        while True:
            char = process.stdout.read(1)
            if not char: # C++ process has exited
                break
            
            # If we hit a space or newline, we have finished reading a Token ID!
            if char == " " or char == "\n":
                if token_accumulator:
                    token_id = int(token_accumulator)
                    
                    # Decode the single token ID and stream it to the console
                    word = enc.decode([token_id])
                    print(word, end="", flush=True)
                    
                    token_accumulator = ""
            else:
                # Accumulate digits of the token ID
                token_accumulator += char
                
    except KeyboardInterrupt:
        print("\n[Generation interrupted by user]")
    finally:
        process.terminate()
        
    print("\n--- Model Generation Finished ---\n")

if __name__ == "__main__":
    generate_text_stream("The sky is ")
