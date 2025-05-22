# evaluate.py
import sys, os, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.decoding_utils import decode_generated_part
from decoding.greedy import greedy_generate
from decoding.beam_search import beam_generate
from decoding.speculative import speculative_generate

# デコード関数をモード名で切り替え
DECODERS = {
    "greedy": greedy_generate,
    "beam": beam_generate,
    "speculative": speculative_generate,
}


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(dataset, cfg, mode):
    draft_model_id = cfg.get("draft_model", None)
    target_model_id = cfg["target_model"]
    k = cfg["decode"].get("k", 4)
    num_beams = cfg["decode"].get("num_beams", 5)
    max_tokens = cfg["decode"].get("max_tokens", 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_id).to(device)

    if mode == "speculative":
        draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
        draft_model = AutoModelForCausalLM.from_pretrained(draft_model_id).to(device)
    else:
        draft_model = draft_tokenizer = None

    decoder = DECODERS[mode]
    correct = 0
    total = len(dataset)

    for item in tqdm(dataset):
        prompt = item["prompt"]
        answer = item["answer"]

        instruction = " Answer with only the final answer."
        full_prompt = prompt + instruction

        input_ids = target_tokenizer(full_prompt, return_tensors="pt").input_ids.to(
            device
        )

        if mode == "greedy":
            output_ids = decoder(
                target_model, target_tokenizer, input_ids, device, max_tokens=max_tokens
            )
        elif mode == "beam":
            output_ids = decoder(
                target_model,
                target_tokenizer,
                input_ids,
                device,
                num_beams=num_beams,
                max_tokens=max_tokens,
            )
        elif mode == "speculative":
            output_ids = decoder(
                draft_model,
                target_model,
                draft_tokenizer,
                input_ids,
                device,
                k=k,
                max_tokens=max_tokens,
            )

        output = decode_generated_part(output_ids, input_ids, target_tokenizer).strip()

        if output.lower() == answer.lower():
            correct += 1

        print(f"[Prompt] {prompt}")
        print(f"[Output] {output}")
        print(f"[Answer] {answer}")
        print("----")

    acc = correct / total
    print(f"Mode: {mode}")
    print(f"Accuracy: {correct} / {total} = {acc:.2%}")


if __name__ == "__main__":
    config = load_config("config/eval.yaml")
    dataset = load_dataset(config["dataset"])
    for mode in config["decode"]["modes"]:
        evaluate(dataset, config, mode)
