import torch

def beam_generate(model, tokenizer, input_ids, device, num_beams=5, max_tokens=50):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.to(device),
            num_beams=num_beams,
            do_sample=False,
            max_new_tokens=max_tokens,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return output_ids
