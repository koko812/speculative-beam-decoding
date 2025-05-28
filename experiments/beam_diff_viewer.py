import sys, os, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ==== トークン差分マーカー ====
def mark_differences(token_lists):
    num_beams = len(token_lists)
    min_len = min(len(t) for t in token_lists)

    marked = [[] for _ in range(num_beams)]

    for i in range(min_len):
        tokens_at_i = [tokens[i] for tokens in token_lists]
        if all(tok == tokens_at_i[0] for tok in tokens_at_i):
            for b in range(num_beams):
                marked[b].append(tokens_at_i[b])
        else:
            for b in range(num_beams):
                marked[b].append(f"**{tokens_at_i[b]}**")

    # 残りのトークン（長いビーム用）
    for b in range(num_beams):
        for i in range(min_len, len(token_lists[b])):
            marked[b].append(f"**{token_lists[b][i]}**")

    return marked


# ==== ファイル読み込み ====
filename = "beam_outputs/The_future_of_AI_is.json"

with open(filename, encoding="utf-8") as f:
    data = json.load(f)

token_lists = [beam["tokens"] for beam in data["beams"]]
marked_tokens = mark_differences(token_lists)

# ==== 表示 ====
print(f"\n🧠 Prompt: {data['prompt']}")
print(f"🔍 分岐トークンを強調表示（**）\n")

for i, tokens in enumerate(marked_tokens):
    print(f"[Beam {i + 1}]")
    print(" ".join(tokens))
    print("-" * 80)
