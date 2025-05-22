from transformers import BatchEncoding
import torch

def render_prompt(prompt, tokenizer, use_system_prompt=False, system_prompt=None, debug=False):
    can_use_chat_template = (
        hasattr(tokenizer, "chat_template")
        and tokenizer.chat_template is not None
    )

    if can_use_chat_template:
        messages = []
        if use_system_prompt and system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if debug:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False)
            print("\n[DEBUG] Input rendered by ChatTemplate:\n" + rendered)

        result = tokenizer.apply_chat_template(messages, return_tensors="pt")
        # ✅ 戻り値が Tensor の場合、自動でラップする
        if isinstance(result, torch.Tensor):
            return BatchEncoding({"input_ids": result})
        return result

    else:
        if debug:
            print("\n[DEBUG] Raw prompt input:\n" + prompt)
        return tokenizer(prompt, return_tensors="pt")
