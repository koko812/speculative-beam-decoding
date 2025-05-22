def render_prompt(prompt, tokenizer, use_system_prompt=False, system_prompt=None, debug=False):

    """
    モデルに応じた入力形式を構築。
    - ChatTemplate（e.g. Mistral、LLaMA2-chatなど）を使う場合は apply_chat_template() を利用。
    - 通常の CausalLM モデルでは tokenizer(prompt) を使う。
    - debug=True のときは実際にモデルに入力されるテキスト（文字列）を出力。

    Parameters:
        prompt (str): ユーザーの入力プロンプト
        tokenizer: Transformers Tokenizer オブジェクト
        use_system_prompt (bool): system prompt を使うかどうか
        system_prompt (str or None): 使用する system prompt の内容
        debug (bool): モデルに渡す入力を出力するかどうか

    Returns:
        transformers.BatchEncoding: input_ids, attention_mask を含む辞書
    """

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

        return tokenizer.apply_chat_template(messages, return_tensors="pt")

    else:
        if debug:
            print("\n[DEBUG] Raw prompt input:\n" + prompt)
        return tokenizer(prompt, return_tensors="pt")
def render_prompt(prompt, tokenizer, use_system_prompt=False, system_prompt=None, debug=False):
    """
    モデルに応じた入力形式を構築。
    ChatTemplate が使える場合のみそれを適用し、なければ通常のトークナイズを行う。
    """
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

        return tokenizer.apply_chat_template(messages, return_tensors="pt")

    else:
        if debug:
            print("\n[DEBUG] Raw prompt input:\n" + prompt)
        return tokenizer(prompt, return_tensors="pt")
