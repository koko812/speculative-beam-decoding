import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.load_models import load_model_and_tokenizer
from models.prompt_renderer import render_prompt  # ✅ 追加
from decoding.speculative import speculative_generate
from decoding.beam_search import beam_generate

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

    draft_model, draft_tokenizer = load_model_and_tokenizer(cfg.draft_model, device)
    target_model, target_tokenizer = load_model_and_tokenizer(cfg.target_model, device)
    print(f"[INFO] Draft model: {cfg.draft_model}")
    print(f"[INFO] Target model: {cfg.target_model}")

    for i, prompt in enumerate(cfg.prompts):
        print(f"\n[Prompt {i+1}] {prompt}")

        # ✅ ここで input_ids を準備（ChatTemplate対応）
        inputs = render_prompt(
            prompt,
            tokenizer=draft_tokenizer if cfg.decode.mode == "speculative" else target_tokenizer,
            use_system_prompt=cfg.chat.use_system_prompt,
            system_prompt=cfg.chat.system_prompt,
            debug=cfg.debug.render_input_text
        )
        input_ids = inputs.input_ids.to(device)

        if cfg.decode.mode == "speculative":
            output = speculative_generate(
                draft_model, target_model, draft_tokenizer,
                input_ids, device,
                k=cfg.decode.k,
                max_tokens=cfg.decode.max_tokens
            )
        else:
            output = beam_generate(
                target_model, target_tokenizer,
                input_ids, device,
                max_tokens=cfg.decode.max_tokens
            )

        print("\n=== Output ===")
        print(output)
        wandb.log({f"prompt_{i+1}": prompt, f"output_{i+1}": output})

if __name__ == "__main__":
    main()
