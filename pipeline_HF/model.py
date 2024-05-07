import torch
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name="mistralai/Mistral-7B-Instruct-v0.2"): 
    # Define the quantization configuration (ref: https://huggingface.co/blog/4bit-transformers-bitsandbytes)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load pre-trained model and tokenizer (might take a while to download model weights)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True, # Trust the model weights from the remote server
        device_map="auto", # Use all RAM from GPU, CPU, disk, in that order (ref: https://huggingface.co/docs/accelerate/en/usage_guides/big_modeling#using--accelerate)
        quantization_config=bnb_config, # Quantize the model using bitsandbytes
        # attn_implementation='flash_attention_2', # Use flash attention 2 (ref: https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one?install=NVIDIA#flashattention-2)
    )
    return model, tokenizer