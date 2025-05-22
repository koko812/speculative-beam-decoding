# models/load_models.py
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # ← GPU対応
    model.eval()
    return model, tokenizer
