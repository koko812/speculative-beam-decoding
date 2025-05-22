# 🧪 Experiment Report: Speculative Decoding + Beam Search with ChatTemplate Models

## 1. 実装内容

本実験は、以下の要素を統合的に組み込んだ**推論効率向上のための比較実験基盤**である：

- 🤖 **Speculative Decoding** の実装（草案モデル + 本番モデルの協調生成）
- 🎯 **Beam Search** の実装（単一モデルによる幅探索型デコード）
- 🔧 **Hydra + W&B** による柔軟な設定管理・実験ログ保存
- 🧩 **ChatTemplate 対応**：`tokenizer.apply_chat_template()` による統一されたチャット入力構成
- ⚙️ `render_prompt()` 関数により CausalLM / Chat 型を統合的に処理
- CLI で切替可能な `system_prompt`, `debug.render_input_text` による入力可視化

---

## 2. 実装の変遷

### ✅ 初期構成
- `distilgpt2` / `gpt2` を使用した **Speculative Decoding** のトイ実装
- `prompt` をコード内でハードコーディング
- `tokenizer(prompt)` によるプレーンなプロンプト生成
- 警告（`attention_mask not set`）が出るも無視

### 🔄 段階的進化
1. `prompt` を `config/default.yaml` に移動 → CLI から差し替え可能に
2. `prompts: [ ... ]` による **複数プロンプトバッチ評価** に対応
3. `render_prompt()` の導入により **ChatTemplate モデルの統合**
4. `debug.render_input_text` による **入力可視化** のトグル実装
5. `input_ids` を run.py 側で統一生成し、`speculative_generate` / `beam_generate` に渡す方式へ移行
6. `tokenizer.chat_template is not None` による安全判定導入
7. `TinyLlama-Chat` → `Gemma 3` → `Qwen2.5-3B-Instruct` へのモデル進化

---

## 3. 気づいたこと・学び

- ✅ ChatTemplate は HuggingFace で徐々に標準化されつつあるが、返り値が `Tensor` のみの場合もあり注意（要 `BatchEncoding` ラップ）
- ⚠️ `hasattr(tokenizer, "apply_chat_template")` だけでは不十分。`tokenizer.chat_template is not None` のチェックが本質
- ✅ `prompt_renderer` 関数にシステムプロンプトやデバッグ機能を統合することで、実験時の挙動を即可視化できた
- ✅ モデルによって `system_prompt` の効果が明確に異なるため、CLIからの切替が非常に有用
- ✅ `TinyLlama` は軽量だが内容は浅く、`Zephyr`, `Gemma`, `Qwen` などの ChatTemplate 系モデルはより自然な出力を示した
- ✅ `speculative decoding` は draft モデルの品質に大きく依存し、低品質だと target model が追いつき fallback 連発になりやすい
- ✅ `torch._dynamo.exc.Unsupported` などの内部エラーはモデルの transformer 実装が新しすぎる場合に起こる（Gemma3）

---

## 4. 実装したデコード手法の概要

### 🧠 Speculative Decoding

複数モデル（draft + target）を用いて、高速性と品質を両立する方法：

1. **草案モデル（draft）** が `k` 個のトークンを予測（高速）
2. **本番モデル（target）** がそのトークンを逐次検証
3. 不一致があれば fallback（target による再生成）
4. 一致したトークンはまとめて採用し高速化

```text
[Prompt] → [草案モデル → kトークン生成] → [targetモデルで検証] → 一致？ → YES → 採用 / NO → fallback
```

- `generate()` の `top_k`, `temperature` により draft の多様性を制御
- 高品質な草案を出せれば、生成速度が最大5倍以上に向上する場合も

---

### 🎯 Beam Search

従来の探索型生成方式：

- 予測の際に「最も確からしい `n` 本のビーム（文脈）」を残しつつ展開
- 最終的に最良のビームを選択

```text
Step 1: 現在のトークンに対し top-n 候補を展開
Step 2: 各候補について再帰的に探索
Step 3: 合計確率が最大の系列を出力
```

- 多様性に乏しいが、決定的で安定した生成に強い
- `num_beams` のチューニングが鍵

---

## 📁 使用モデル一覧と所感

| モデル | 使用可否 | ChatTemplate | 所感 |
|--------|----------|--------------|------|
| distilgpt2 / gpt2 | ✅ | ❌ | トイ実験には軽くて良いが出力が単調 |
| TinyLlama-Chat | ✅ | ✅ | 超軽量・即応答だが知識は浅め |
| Gemma 3 4B | ❌（エラー） | ✅ | ChatTemplateありだが Transformers 側未対応あり |
| Zephyr 1.3B | ✅ | ✅ | 高品質・軽量で実用向き |
| Qwen2.5-3B-Instruct | ✅ | ✅ | 多言語＋ChatTemplate対応、推奨モデル |

---

以上が、本実験で得られた構成・知見のまとめです。今後はモデル比較、system prompt制御、出力品質評価（BLEU, perplexity）などに拡張できます。
