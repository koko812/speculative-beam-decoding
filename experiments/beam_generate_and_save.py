import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.prompt_renderer import render_prompt

# ==== 設定 ====
model_id = "Qwen/Qwen2.5-3B-Instruct"
prompt = "The future of AI is"
num_beams = 5
max_new_tokens = 50
output_dir = "beam_outputs"

# ==== モデル & トークナイザ読み込み ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# ==== プロンプトレンダリング ====
inputs = render_prompt(
    prompt,
    tokenizer=tokenizer,
    use_chat_template=True,
    use_system_prompt=True,
    system_prompt="You are a helpful assistant.",
    debug=True,
)
input_ids = inputs.input_ids.to(device)

# ==== 生成 ====
outputs = model.generate(
    input_ids=input_ids,
    num_beams=num_beams,
    num_return_sequences=num_beams,
    max_new_tokens=max_new_tokens,
    return_dict_in_generate=True,
    output_scores=True,
)

# ==== ビームごとのトークン・確率・ロジットを計算 ====
beam_details = []

for b in range(num_beams):
    gen_ids = outputs.sequences[b][input_ids.shape[-1]:]
    tokens = tokenizer.convert_ids_to_tokens(gen_ids)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    logit_list = []
    logprob_list = []

    for t, logits in enumerate(outputs.scores):
        log_probs = F.log_softmax(logits, dim=-1)
        token_id = outputs.sequences[b][input_ids.shape[-1] + t].item()
        logit_list.append(logits[b][token_id].item())
        logprob_list.append(log_probs[b][token_id].item())

    beam_details.append({
        "text": text,
        "tokens": tokens,
        "ids": gen_ids.tolist(),
        "logits": logit_list,
        "logprobs": logprob_list,
        "logprob_total": sum(logprob_list)
    })

# ==== 保存 ====
result = {
    "model": model_id,
    "prompt": prompt,
    "num_beams": num_beams,
    "max_new_tokens": max_new_tokens,
    "beams": beam_details
}

os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"{prompt[:30].replace(' ', '_')}.json")
with open(filename, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"✅ 結果を保存しました → {filename}")
