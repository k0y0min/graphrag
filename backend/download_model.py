import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    model_id = "Qwen/Qwen3-0.6B"
    token = os.getenv("HF_TOKEN")
    
    print(f"Checking for model {model_id}...")
    if token:
        print("HF_TOKEN found in environment.")
    else:
        print("WARNING: HF_TOKEN not found in environment. Public models may still work.")

    try:
        # Download tokenizer
        AutoTokenizer.from_pretrained(model_id, cache_dir="./models", token=token)
        # Download model
        AutoModelForCausalLM.from_pretrained(model_id, cache_dir="./models", token=token)
        print("Model download complete.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_model()
