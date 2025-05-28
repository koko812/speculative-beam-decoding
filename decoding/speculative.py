import torch
from transformers import LogitsProcessorList, MinLengthLogitsProcessor
from models.prompt_renderer import render_prompt
import logging


def speculative_generate(
    draft_model, target_model, tokenizer, input_ids, device, k=4, max_tokens=50, use_chat_template=False
):
    logger = logging.getLogger("speculative_debug")
    handler = logging.FileHandler("logs/speculative_debug.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    rendered = render_prompt(
        tokenizer.decode(input_ids[0], skip_special_tokens=True),
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
        use_system_prompt=use_chat_template,
        system_prompt="You are a helpful assistant." if use_chat_template else None,
        debug=True
    )
    generated = rendered.input_ids.to(device)

    init_prompt = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("🔰 Initial prompt:", init_prompt)
    logger.info(f"🔰 Initial prompt: {init_prompt}")

    while generated.shape[1] < max_tokens:
        with torch.no_grad():
            draft_outputs = draft_model.generate(
                generated,
                max_new_tokens=k,
                do_sample=False,
                top_k=1,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        draft_tokens = draft_outputs[:, generated.shape[1]:]
        draft_text = tokenizer.decode(draft_tokens[0], skip_special_tokens=True)
        print("\n✏️ Draft tokens:", draft_text)
        logger.info(f"\n✏️ Draft tokens: {draft_text}")

        accepted_tokens = []

        for i in range(draft_tokens.shape[1]):
            current_input = torch.cat([generated, draft_tokens[:, :i]], dim=-1)
            with torch.no_grad():
                target_outputs = target_model(current_input)
                next_token_logits = target_outputs.logits[:, -1, :]
                predicted_token = torch.argmax(next_token_logits, dim=-1)

            draft_tok = draft_tokens[:, i].item()
            pred_tok = predicted_token.item()
            decoded_draft = tokenizer.decode([draft_tok])
            decoded_pred = tokenizer.decode([pred_tok])

            msg = f"  🧪 Step {i + 1}: Draft token = '{decoded_draft}', Target predicted = '{decoded_pred}'"
            print(msg)
            logger.info(msg)

            if pred_tok == draft_tok:
                accepted_tokens.append(draft_tokens[:, i])
            else:
                print("  ❌ Mismatch → fallback")
                logger.info("  ❌ Mismatch → fallback")
                with torch.no_grad():
                    fallback = target_model.generate(
                        current_input,
                        max_new_tokens=1,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated = torch.cat([generated, fallback[:, -1:]], dim=-1)
                if fallback[0, -1].item() == tokenizer.eos_token_id:
                    print("🛑 Target model generated <eos>. Stopping.")
                    logger.info("🛑 Target model generated <eos>. Stopping.")
                    break
                break
        else:
            if accepted_tokens:
                accepted = torch.cat(accepted_tokens, dim=-1).unsqueeze(0)
                generated = torch.cat([generated, accepted], dim=-1)
                if tokenizer.eos_token_id in accepted:
                    print("🛑 Draft tokens included <eos>. Stopping.")
                    logger.info("🛑 Draft tokens included <eos>. Stopping.")
                    break
                accepted_text = tokenizer.decode(accepted[0], skip_special_tokens=True)
                print("  ✅ All draft tokens accepted:", accepted_text)
                logger.info(f"  ✅ All draft tokens accepted: {accepted_text}")

    final_output = tokenizer.decode(generated[0], skip_special_tokens=True)
    print("\n✅ Final generated:", final_output)
    logger.info(f"\n✅ Final generated: {final_output}\n{'=' * 60}\n")
    return generated
