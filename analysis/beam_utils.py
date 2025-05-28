def print_generation_result(output_ids, input_ids, tokenizer, num_beams):
    gen_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"[num_beams={num_beams}] → {gen_text}")
