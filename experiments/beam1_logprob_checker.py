import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-3B-Instruct"
prompt = "The future of AI is"

# モデル読み込み
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# プロンプト処理（ChatTemplate）
from models.prompt_renderer import render_prompt
inputs = render_prompt(
    prompt,
    tokenizer=tokenizer,
    use_chat_template=True,
    use_system_prompt=True,
    system_prompt="You are a helpful assistant.",
    debug=True,
)
input_ids = inputs.input_ids.to(model.device)

# 生成（greedy相当）
outputs = model.generate(
    input_ids=input_ids,
    num_beams=1,
    num_return_sequences=1,
    max_new_tokens=50,
    return_dict_in_generate=True,
    output_scores=True,
    output_hidden_states=False,
)

gen_ids = outputs.sequences[0][input_ids.shape[-1]:]
tokens = tokenizer.convert_ids_to_tokens(gen_ids)

print("\n🧪 Beam=1 での検証結果\n")
for t, logits in enumerate(outputs.scores):
    log_probs = F.log_softmax(logits, dim=-1)
    token_id = gen_ids[t].item()
    token = tokens[t]
    logit = logits[0][token_id].item()
    log_prob = log_probs[0][token_id].item()

    print(f"{t:2d}: Token: {token:>12} | ID: {token_id:5d} | Logit: {logit:>7.2f} | LogProb: {log_prob:>7.2f}")
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-3B-Instruct"
prompt = "The future of AI is"

# モデル読み込み
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# プロンプト処理（ChatTemplate）
from models.prompt_renderer import render_prompt
inputs = render_prompt(
    prompt,
    tokenizer=tokenizer,
    use_chat_template=True,
    use_system_prompt=True,
    system_prompt="You are a helpful assistant.",
    debug=True,
)
input_ids = inputs.input_ids.to(model.device)

# 生成（greedy相当）
outputs = model.generate(
    input_ids=input_ids,
    num_beams=1,
    num_return_sequences=1,
    max_new_tokens=50,
    return_dict_in_generate=True,
    output_scores=True,
    output_hidden_states=False,
)

gen_ids = outputs.sequences[0][input_ids.shape[-1]:]
tokens = tokenizer.convert_ids_to_tokens(gen_ids)

print("\n🧪 Beam=1 での検証結果\n")
for t, logits in enumerate(outputs.scores):
    log_probs = F.log_softmax(logits, dim=-1)
    token_id = gen_ids[t].item()
    token = tokens[t]
    logit = logits[0][token_id].item()
    log_prob = log_probs[0][token_id].item()

    print(f"{t:2d}: Token: {token:>12} | ID: {token_id:5d} | Logit: {logit:>7.2f} | LogProb: {log_prob:>7.2f}")
