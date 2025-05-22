import json
from datetime import datetime
from pathlib import Path

def log_output(
    mode,
    prompt_id,
    prompt,
    output,
    target_model,
    draft_model=None,
    k=None,
    max_tokens=None,
    num_beams=None,
    do_sample=None,
    outpath="logs/outputs.jsonl",
    enabled=True,
    generated_tokens=None,
    tokens_per_sec=None
):
    if not enabled:
        return

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "a", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "output": output,
            "timestamp": datetime.now().isoformat(),
            "model": {
                "target": target_model,
                "draft": draft_model
            },
            "decode_params": {
                "k": k,
                "max_tokens": max_tokens,
                "num_beams": num_beams,
                "do_sample": do_sample
            },
            "metrics": {
                "generated_tokens": generated_tokens,
                "tokens_per_sec": tokens_per_sec
            }
        }, f, ensure_ascii=False)
        f.write("\n")
