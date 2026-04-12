import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import gc

def benchmark(name, gen_func, input_ids, max_new_tokens=100):
    print(f"--- Running {name} ---")
    # Warmup
    _ = gen_func(input_ids, max_new_tokens=10)
    
    # Measure
    start_time = time.time()
    output = gen_func(input_ids, max_new_tokens=max_new_tokens)
    end_time = time.time()
    
    if isinstance(output, tuple):
        output_ids = output[0]
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
    
    model_id = "google/medgemma-4b-it"
    
    print(f"Loading Med-Gemma model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load the text-generation model in float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Medical prompt
    prompt = "Explain the differential diagnosis for a patient presenting with acute chest pain, including key distinguishing features between cardiac and non-cardiac causes."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"Input tokens: {input_ids.shape[1]}")
    
    max_new_tokens = 100
    
    # 1. Vanilla LLM (Med-Gemma baseline)
    def gen_vanilla(ids, max_new_tokens):
        return model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    
    tps_vanilla, output = benchmark("Med-Gemma 4B Vanilla LLM", gen_vanilla, input_ids, max_new_tokens)
    
    # Decode and print sample output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n--- Sample Output ---")
    print(generated_text[:500])
    
    print(f"\n=== Med-Gemma Benchmark Summary ===")
    print(f"Med-Gemma 4B Vanilla LLM: {tps_vanilla:.2f} tokens/sec")

if __name__ == "__main__":
    main()
