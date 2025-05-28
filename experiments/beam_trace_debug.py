from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


model_id = "facebook/opt-1.3b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

#prompt = "The future of AI is"
prompt = "The future of AI is expected to be"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    num_beams=4,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=50,
)


sequence = outputs.sequences[0]  # 最終出力シーケンス（Beam 0 のみとは限らない）
token_ids = sequence[inputs["input_ids"].shape[-1]:]  # 生成されたトークンのみ

print(f"Total tokens in sequence: {len(token_ids)}")
print(f"Available scores:         {len(outputs.scores)}")
print(f"Available beam_indices:   {len(outputs.beam_indices)}")



logprobs = []
prev_beam = 0  # 初期ビーム（仮定）

print("\n📊 Beam Trace per Token with Beam Switching Info:")
for step, (token_id, score_logits, beam_idx) in enumerate(zip(token_ids, outputs.scores, outputs.beam_indices)):
    probs = F.log_softmax(score_logits, dim=-1)
    token_logprobs = probs[:, token_id.item()]  # [num_beams]

    # 最も尤もらしいビーム（このトークンを出力したと思われるビーム）
    current_beam = torch.argmax(token_logprobs).item()
    logprob = token_logprobs[current_beam].item()
    token_str = tokenizer.decode([token_id])

    if current_beam != prev_beam:
        switch_info = f"→ switched from Beam {prev_beam}"
    else:
        switch_info = ""

    print(f"Step {step+1:2}: Token: {token_str:15} | LogProb: {logprob:7.2f} | From Beam {current_beam} {switch_info}")
    prev_beam = current_beam