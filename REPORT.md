# RPORT Log: Beam Search Trace Debugging (2025/5/28)

## 概要

Beam search 生成中の各トークンにおいて、どのビームから来たのかをトレースする仕組みを実験した。

## 試行内容と過程の詳細

### 1. 初期構成

* `Qwen/Qwen2.5-3B-Instruct` を使って beam search を実施。
* `output_scores=True` によって `outputs.scores` を取得。
* 生成された各ビーム出力を `outputs.sequences` からトークンごとに解析。

### 2. Beam のトレースを試みる

* `beam_indices` によってどのビームがどこから来たかをトレースしようとしたが、`Qwen` モデルでは `beam_indices` が取得できなかった。
* ログに `AttributeError: 'GenerateDecoderOnlyOutput' object has no attribute 'beam_indices'` が出力される。

### 3. モデル変更の検討

* beam\_indices をきちんと返すモデルを探す。
* `GPT2` を使ってみたが、`beam_indices` の shape が (1, sequence\_length) のままで、生成ステップごとの変化が見られず。

### 4. `Qwen1.5-1.8B` に変更して再試行

* `output_scores=True` に加え、`beam_indices` が得られた。
* ただし `beam_indices` の長さが 1 ステップ分しかなかった。
* 生成されたシーケンス長（50）と `outputs.scores` の長さ（50）に対して、`beam_indices` は 1 ステップしか提供されない。
* 他のモデル（`Qwen3-1.7B`, `GPT2`）でも同様の結果。

### 5. 擬似的なトレース方法

* `beam_indices` が壊れている可能性を考慮し、`scores` を用いた擬似的なトレース手法に切り替え。
* 各ステップの `score_logits` に対して `log_softmax` を取る。
* 各ビームにおける該当トークンの log probability を比較し、最も尤もらしいビームを `argmax` により推定。

```python
for step, (token_id, score_logits) in enumerate(zip(token_ids, outputs.scores)):
    probs = F.log_softmax(score_logits, dim=-1)
    token_logprobs = probs[:, token_id.item()]
    best_beam = torch.argmax(token_logprobs).item()
```

### 6. 擬似的なトレースの限界

* `scores` に含まれるのは「そのステップでの候補ビーム」に限定される。
* トレースには backpointer（遷移元情報）が必要だが、それは保存されていない。
* よって、最終的な出力に至るまでにどんなビーム遷移が起きていたのかを完全に復元することはできない。
* 結果として、現在の実装では「一番尤もらしい遷移」を毎ステップで局所的に推定しているにすぎない。

## 結論と学び

* 多くのモデルでは `beam_indices` が正しく出力されない、またはステップ単位での追跡ができないケースがある。
* その代替として、`scores` を使って局所的に beam の遷移を「推定」する方法が有効。
* しかしそれは真のトレースではなく、「擬似的な復元」にすぎない。
* beam search の内部状態（beam candidates, backpointer）を追うには、transformers ライブラリ自体の改変が必要になる可能性が高い。

## 今後の展望

* Beam 遷移の完全なトレースには、transformers の内部ロジックをフックする方法も検討する。
* あるいは、小さな独自モデルを用いて beam 遷移を明示的に log 出力する自前生成ループを組むことも一案。
* 複数モデル（例：大・小モデル）の組み合わせによる協調生成に向けて、まずはビームの挙動をより詳細に可視化するツールを作成していく。

</br>
</br>
</br>
</br>
</br>

# 🧪 Evaluate QA Accuracy + ChatTemplate Control（v3 - 2025-05-23）

## 🧭 概要

本フェーズでは、speculative decoding における draft モデルへの ChatTemplate 適用を**明示的に制御可能**にする機構を追加し、精度向上と挙動安定性の改善を目指した。これにより、ChatTemplate をもたない draft モデル使用時の異常出力を防止し、正答率が初めて 0% を脱する結果が得られた。

---

## ✅ 改善内容まとめ

### 1. `speculative_generate()` にオプション追加

* `use_chat_template=False` をデフォルト引数として導入
* `render_prompt()` の呼び出し時に chat template 適用可否を明示指定
* ChatTemplate 対応・非対応モデル間での挙動差を安全に吸収

### 2. `prompt_renderer.py` の強化

* `tokenizer.chat_template is not None` による安全な template 判定を維持
* `debug=True` 指定で、ChatTemplate 適用時にレンダリング結果を標準出力に明示表示

### 3. `evaluate.py` 側の強化

* `eval.yaml` に `use_chat_template` オプションを追加
* `evaluate()` → `speculative_generate()` 呼び出し時にそのまま反映

```yaml
# eval.yaml の例
decode:
  modes: ["speculative"]
  k: 4
  max_tokens: 50
  use_chat_template: true  # ← これを切り替えるだけで挙動変化
```

---

## 📊 精度改善の傾向

### ❌ ChatTemplate 適用時の誤出力の具体例

以下は ChatTemplate を draft model に適用したことで観測された誤出力例：

* **Q: What is the capital of France?**

  * ❌ 出力: `ParisHuman: Write a title for this article:`
  * 🔍 問題点: ChatTemplate によってプロンプトテンプレート（例：Human: や Assistant:）がそのまま出力に混入してしまい、構造が破綻する

* **Q: What is H2O?**

  * ❌ 出力: `The answer is: H2O is a molecule consisting of...`
  * 🔍 問題点: 指定された形式（"only the final answer"）に反し、冗長な解説が追加されてしまう

* **Q: Who wrote 1984?**

  * ✅ 修正後出力: `George Orwell`
  * 🔍 ChatTemplate を使わなかったことで、冗長な補足文が排除され、簡潔に正答が生成された

* ChatTemplate を適用した結果、プロンプトテンプレートそのものが出力文に混入し、正答の直後に `Human:` や `You are...` などの文言が続くなど不自然な構造が多発したため、
  本実験では最終的に ChatTemplate は **使用しない設定** が採用された。

* Draft モデルと Target モデルの構文的整合性が増し、Accept トークンが連続する事例が確認された

* 正答率も初めて 0% を脱し、意味のある比較が可能な状態に到達

---

## 📌 反映ファイル一覧

| ファイル                        | 修正点                                    |
| --------------------------- | -------------------------------------- |
| `decoding/speculative.py`   | `use_chat_template` 引数の追加と制御処理実装       |
| `models/prompt_renderer.py` | ChatTemplate 適用/非適用の切替・デバッグ出力          |
| `experiments/evaluate.py`   | config から `use_chat_template` を渡す処理の追加 |

---

## 📝 所感と次ステップ

* ChatTemplate の有無はモデル選定と密接に関わるため、今後は `AutoTokenizer` のメタデータをより詳細に検査する必要がある
* draft モデルが Instruct モデルでなくとも ChatTemplate を誤って適用してしまう問題は本修正で解決
* 次ステップでは `target` モデル側にも ChatTemplate を適用した場合の挙動比較、および `system_prompt` の影響分析を行う予定

---

これにより、実験の信頼性と再現性が一段と向上した。

</br>
</br>
</br>
</br>
</br>

# 🧪 Evaluate QA Accuracy に関する進捗ログ（v2 - 2025-05-23）

## 🧭 概要

本フェーズでは、speculative decoding の基本動作を確認・可視化するためのベースライン評価を整備し、greedy / beam / speculative 各方式の出力を QA タスク上で比較検証した。特に、speculative decoding における draft モデルと target モデルの挙動差が精度に与える影響を観察することに主眼を置いた。

---

## ✅ 評価スクリプトの拡張

* `evaluate.py` において YAML 設定ファイルから複数モード（greedy, beam, speculative）を同時評価可能な構造へ拡張。
* `datasets/qa_dataset.json` を導入し、30問の QA に対して自動精度評価が可能に。
* 出力は各プロンプトに対して `[Prompt]`, `[Output]`, `[Answer]` の形式でログ表示・評価される。

---

## 📊 精度結果（初回ベンチマーク）

| モード         | 正答数 | 正答率   |
| ----------- | --- | ----- |
| greedy      | 12  | 40.0% |
| beam (5本)   | 11  | 36.7% |
| speculative | 0   | 0.0%  |

> speculative モードの精度が極端に低い理由を解析するため、内部の token accept ロジックを可視化するデバッグロガーを導入。

---

## 🔍 Speculative Decoding の問題点

* draft モデルの生成出力が、ターゲットモデルの予測と一致せずほぼ毎回 fallback。
* draft 側の設定が `do_sample=True`, `top_k=50`, `temperature=0.7` とランダム性が強すぎ、QA タスクでは不適。
* draft 出力に ChatTemplate 系の文言（"You are", "Human:", etc.）が混入しやすく、ターゲットの予測と構文的に乖離。

---

## 🔧 改良と今後の展望

### 現在の改善内容

* `speculative_generate()` にトークンごとのログ記録と fallback 検出機能を追加。
* `evaluate.py` をマルチモード対応に変更し、mode 毎に独立した出力を得られるように調整。
* `speculative_generate()` の戻り値をトークン ID に統一し、外部でデコード処理する構成へ変更。

### 今後の技術的検討事項

1. **accept 判定を softmax 比較ベースへ（p\_target / p\_draft）**
2. **Gumbel Sampling + Top-k 抽出による安定した draft 提案の実装**
3. **fallback 時の再サンプリングを精度保証付きで実行する構造化**
4. **既存実装（speculative-decoding-main）との統合検証**

---

## 📂 関連ファイル

| ファイルパス                       | 内容                              |
| ---------------------------- | ------------------------------- |
| `logs/speculative_debug.log` | デバッグ用ステップログ出力                   |
| `datasets/qa_dataset.json`   | QA タスクデータ（30問）                  |
| `experiments/evaluate.py`    | ベンチマーク評価スクリプト（multi-mode対応）     |
| `decoding/speculative.py`    | 改良版 `speculative_generate()` 実装 |

---

次フェーズでは、正答率向上と fallback 削減の両立を目指し、accept ロジックと draft 生成の精度制御に焦点を当てていく予定である。

</br>
</br>
</br>
</br>
</br>


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