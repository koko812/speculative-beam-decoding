def decode_generated_part(output_ids, input_ids, tokenizer):
    """
    モデルの出力トークン列から、プロンプト部分を除いた生成トークンをデコードする。

    Parametersu:
        output_ids (Tensor): generate() によるモデルの出力（1行分）
        input_ids (Tensor): 入力時に与えたプロンプトのトークン列
        tokenizer (PreTrainedTokenizer): 使用中の tokenizer インスタンス

    Returns:
        str: モデルが生成したテキスト部分のみ（プロンプトは含まない）
    """
    generated_ids = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
