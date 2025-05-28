import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoding.beam_search import beam_generate
from analysis.beam_utils import print_generation_result
from models.prompt_renderer import render_prompt


@hydra.main(version_base=None, config_path="../config", config_name="beam_analysis")
def main(cfg: DictConfig):
    print("\n📋 Beam Search Analysis Config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    model = AutoModelForCausalLM.from_pretrained(cfg.target_model).to(device)

    for prompt in cfg.prompts:
        print(f"\n🧠 Prompt: {prompt}")

        # ✅ ChatTemplate + system_prompt 対応
        inputs = render_prompt(
            prompt,
            tokenizer=tokenizer,
            use_system_prompt=cfg.chat.use_system_prompt,
            system_prompt=cfg.chat.system_prompt,
            debug=True,
            use_chat_template=cfg.decode.use_chat_template,
        )
        input_ids = inputs.input_ids.to(device)

        for n in cfg.decode.beam_widths:
            output_ids = beam_generate(
                model, tokenizer, input_ids, device,
                num_beams=n,
                max_tokens=cfg.decode.max_tokens
            )
            print_generation_result(output_ids, input_ids, tokenizer, num_beams=n)


if __name__ == "__main__":
    main()
