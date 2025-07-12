from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Qwen/Qwen3-8B-AWQ"
save_dir = "/workspace/models/Qwen3-8B-AWQ"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Save the model and tokenizer locally
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
