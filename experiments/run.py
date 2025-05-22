import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch

from models.load_models import load_model_and_tokenizer
from models.prompt_renderer import render_prompt
from decoding.speculative import speculative_generate
from decoding.beam_search import beam_generate
from decoding.greedy import greedy_generate
from utils.logger import log_output
from utils.decoding_utils import decode_generated_part

@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    print("\n=== Experiment Config ===")
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.project_name,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    target_model, target_tokenizer = load_model_and_tokenizer(cfg.target_model, device)

    draft_model = draft_tokenizer = None
    if cfg.decode.mode in ("speculative", "hybrid"):
        draft_model, draft_tokenizer = load_model_and_tokenizer(cfg.draft_model, device)

    for i, prompt in enumerate(cfg.prompts):
        print(f"\n[Prompt {i+1}] {prompt}")

        tokenizer = draft_tokenizer if cfg.decode.mode == "speculative" else target_tokenizer

        inputs = render_prompt(
            prompt,
            tokenizer=tokenizer,
            use_system_prompt=cfg.chat.use_system_prompt,
            system_prompt=cfg.chat.system_prompt,
            debug=cfg.debug.render_input_text
        )
        input_ids = inputs.input_ids.to(device)

        # ベンチマーク開始
        start_time = time.time()

        if cfg.decode.mode == "speculative":
            output_ids = speculative_generate(
                draft_model, target_model, draft_tokenizer,
                input_ids, device,
                k=cfg.decode.k,
                max_tokens=cfg.decode.max_tokens
            )
        elif cfg.decode.mode == "beam":
            output_ids = beam_generate(
                target_model, target_tokenizer,
                input_ids, device,
                num_beams=cfg.decode.num_beams,
                max_tokens=cfg.decode.max_tokens
            )
        elif cfg.decode.mode == "greedy":
            output_ids = greedy_generate(
                target_model, target_tokenizer,
                input_ids, device,
                max_tokens=cfg.decode.max_tokens
            )
        else:
            raise ValueError(f"Unknown decode mode: {cfg.decode.mode}")

        elapsed_time = time.time() - start_time
        generated_tokens = output_ids.shape[1] - input_ids.shape[1]
        tokens_per_sec = generated_tokens / elapsed_time if elapsed_time > 0 else 0.0


        cleaned_output = decode_generated_part(output_ids, input_ids, tokenizer)

        print("\n=== Output ===")
        print(cleaned_output)

        wandb.log({f"prompt_{i+1}": prompt, f"output_{i+1}": cleaned_output})

        log_output(
            mode=cfg.decode.mode,
            prompt_id=i + 1,
            prompt=prompt,
            output=cleaned_output,
            target_model=cfg.target_model,
            draft_model=cfg.draft_model if cfg.decode.mode in ("speculative", "hybrid") else None,
            k=cfg.decode.k if cfg.decode.mode in ("speculative", "hybrid") else None,
            max_tokens=cfg.decode.max_tokens,
            num_beams=cfg.decode.num_beams if cfg.decode.mode == "beam" else None,
            do_sample=(cfg.decode.mode == "speculative"),
            outpath=cfg.logging.path,
            enabled=cfg.logging.enabled,
            generated_tokens=generated_tokens,
            tokens_per_sec=tokens_per_sec

        )

if __name__ == "__main__":
    main()
