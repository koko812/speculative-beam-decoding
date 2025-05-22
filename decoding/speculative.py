import torch
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from models.prompt_renderer import render_prompt

def speculative_generate(draft_model, target_model, tokenizer, input_ids, device, k=4, max_tokens=50):
    generated = input_ids.clone()

    while generated.shape[1] < max_tokens:
        with torch.no_grad():
            draft_outputs = draft_model.generate(
                generated,
                max_new_tokens=k,
                do_sample=True,
                top_k=50,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        draft_tokens = draft_outputs[:, generated.shape[1]:]
        accepted_tokens = []

        for i in range(draft_tokens.shape[1]):
            current_input = torch.cat([generated, draft_tokens[:, :i]], dim=-1)
            with torch.no_grad():
                target_outputs = target_model(current_input)
                next_token_logits = target_outputs.logits[:, -1, :]
                predicted_token = torch.argmax(next_token_logits, dim=-1)

            if predicted_token.item() == draft_tokens[:, i].item():
                accepted_tokens.append(draft_tokens[:, i])
            else:
                with torch.no_grad():
                    fallback = target_model.generate(
                        current_input,
                        max_new_tokens=1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                generated = fallback
                break
        else:
            if accepted_tokens:
                accepted = torch.cat(accepted_tokens, dim=-1).unsqueeze(0)
                generated = torch.cat([generated, accepted], dim=-1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

