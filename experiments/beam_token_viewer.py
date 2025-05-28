import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.prompt_renderer import render_prompt

model_id = "Qwen/Qwen2.5-3B-Instruct"
prompt = "The future of AI is"
num_beams = 3
max_new_tokens = 50

# Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Render prompt using ChatTemplate
inputs = render_prompt(
    prompt,
    tokenizer=tokenizer,
    use_chat_template=True,
    use_system_prompt=True,
    system_prompt="You are a helpful assistant.",
    debug=True,
)

input_ids = inputs.input_ids.to(device)

# Generate with beam search (return all beams)
outputs = model.generate(
    input_ids=input_ids,
    num_beams=num_beams,
    num_return_sequences=num_beams,
    max_new_tokens=max_new_tokens,
    return_dict_in_generate=True,
    output_scores=True,
    #output_beam_indices=True,  # ←これ！
)


"""
outputs = model.generate(
    input_ids=input_ids,
    do_sample=False,            # greedy
    num_beams=1,                # beam off
    max_new_tokens=max_new_tokens,
    return_dict_in_generate=True,
    output_scores=True
)"""

sequence = outputs.sequences[0]  # 1つ目のシーケンス（通常 Beam 0）

logprobs = []
token_ids = sequence[input_ids.shape[-1]:]  # 生成された部分

for step, (token_id, score_logits) in enumerate(zip(token_ids, outputs.scores)):
    # logits は [num_beams x vocab_size]
    probs = F.log_softmax(score_logits, dim=-1)
    
    # ここでは Beam 0 から出力されたものと仮定（より正確には backtrack 必要）
    # 各 beam に対して token_id の logprob を調べる
    token_logprobs = probs[:, token_id.item()]  # shape: [num_beams]
    
    best_beam = torch.argmax(token_logprobs).item()
    logprob = token_logprobs[best_beam].item()
    
    token_str = tokenizer.decode([token_id])
    print(f"Step {step+1}: Token: {token_str:15} | LogProb: {logprob:7.2f} | From Beam {best_beam}")