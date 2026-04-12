import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# We will need to add EAGLE to the path if we are outside of it, but since we cloned it here,
# we can just append it to sys.path
sys.path.append("./EAGLE")
from eagle.model.ea_model import EaModel

def benchmark(name, gen_func, input_ids, max_new_tokens=100):
    print(f"--- Running {name} ---")
    # Warmup
    _ = gen_func(input_ids, max_new_tokens=10)
    
    # Measure
    start_time = time.time()
    output = gen_func(input_ids, max_new_tokens=max_new_tokens)
    end_time = time.time()
    
    # The output might include input_ids, so we count only new tokens
    # For EAGLE it might return full sequence, for HF generate it usually does too
    if isinstance(output, tuple):
        # EAGLE eagenerate returns output_ids and maybe other things
        output_ids = output[0] if isinstance(output, tuple) else output
    else:
        output_ids = output
        
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    elapsed = end_time - start_time
    tokens_per_sec = num_new_tokens / elapsed
    print(f"Generated {num_new_tokens} tokens in {elapsed:.2f} seconds ({tokens_per_sec:.2f} tokens/sec)")
    return tokens_per_sec, output_ids

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    base_model_id = "Qwen/Qwen3-1.7B"
    draft_model_id = "Qwen/Qwen2.5-0.5B" # fallback draft model
    eagle_model_id = "AngelSlim/Qwen3-1.7B_eagle3" # EAGLE-3 official weight
    
    print(f"Loading Base Model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # Load base model for vanilla and vanilla spec
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            torch_dtype=torch.float16,
            device_map=device
        )
    except Exception as e:
        print(f"Failed to load base model: {e}")
        return

    print(f"Loading Draft Model for Vanilla Speculative: {draft_model_id}")
    try:
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_id,
            torch_dtype=torch.float16,
            device_map=device
        )
    except Exception as e:
        print(f"Failed to load draft model: {e}")
        draft_model = None

    prompt = "Explain the theory of relativity in detail."
    # Qwen uses specific chat templates, but for benchmark we can just use normal encoding
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    max_new_tokens = 100
    results = {}
    
    # 1. Vanilla LLM
    def gen_vanilla(ids, max_new_tokens):
        return base_model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    
    tps_vanilla, _ = benchmark("Vanilla LLM", gen_vanilla, input_ids, max_new_tokens)
    results["Vanilla LLM"] = tps_vanilla
    
    # 2. Vanilla Speculative Decoding
    if draft_model:
        def gen_spec(ids, max_new_tokens):
            return base_model.generate(
                ids, 
                assistant_model=draft_model, 
                max_new_tokens=max_new_tokens, 
                do_sample=False
            )
        tps_spec, _ = benchmark("Vanilla Speculative Decoding", gen_spec, input_ids, max_new_tokens)
        results["Vanilla Speculative"] = tps_spec
    
    # Clear memory before loading EAGLE to prevent OOM
    del base_model
    if draft_model:
        del draft_model
    import gc
    gc.collect()
    torch.mps.empty_cache() if device == "mps" else None

    # 3. EAGLE 
    # Use load_in_4bit or 8bit if needed, but Mac MPS doesn't support bitsandbytes.
    # So we load in float16.
    print(f"Loading EAGLE model: Base={base_model_id}, EAGLE={eagle_model_id}")
    ea_model = EaModel.from_pretrained(
        base_model_path=base_model_id,
        ea_model_path=eagle_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
        total_token=-1,
        use_eagle3=True # It's an EAGLE-3 model
    )
    ea_model.eval()
    
    def gen_eagle(ids, max_new_tokens):
        return ea_model.eagenerate(ids, max_new_tokens=max_new_tokens, temperature=0.0) # Greedy
    
    tps_eagle, _ = benchmark("EAGLE-3 Decoding", gen_eagle, input_ids, max_new_tokens)
    results["EAGLE-3"] = tps_eagle

    print("\n--- Benchmark Summary ---")
    for name, tps in results.items():
        print(f"{name}: {tps:.2f} tokens/sec")

if __name__ == "__main__":
    main()
