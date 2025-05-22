import torch

def greedy_generate(model, tokenizer, input_ids, device, max_tokens=50):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(device),
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
    return output_ids
