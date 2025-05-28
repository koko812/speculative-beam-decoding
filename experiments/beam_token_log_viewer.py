import json
import os

# ==== 設定 ====
filename = "beam_outputs/The_future_of_AI_is.json"

with open(filename, encoding="utf-8") as f:
    data = json.load(f)

print(f"\n🧠 Prompt: {data['prompt']}")
print(f"🔢 Model: {data['model']}")
print(f"🔍 Beam Width: {data['num_beams']}\n")

for i, beam in enumerate(data["beams"]):
    print(f"[Beam {i + 1}]")
    print(f"Text: {beam['text']}")
    print(f"Total LogProb: {beam['logprob_total']:.2f}")
    print("-" * 60)
    for t_idx, (tok, logit, logprob) in enumerate(
        zip(beam["tokens"], beam["logits"], beam["logprobs"])
    ):
        print(
            f" {t_idx:2d}: {tok:>12} | Logit: {logit:>7.2f} | LogProb: {logprob:>7.2f}"
        )
    print("=" * 60)
