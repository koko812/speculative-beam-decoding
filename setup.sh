#!/bin/bash

mkdir -p config models decoding experiments

# config/default.yaml
cat > config/default.yaml <<EOF
project_name: specbeam-study
draft_model: distilgpt2
target_model: gpt2
decode:
  mode: speculative  # or 'beam'
  k: 4
  max_tokens: 50
EOF

# requirements.txt
cat > requirements.txt <<EOF
transformers>=4.35.0
torch
omegaconf
hydra-core
wandb
EOF

echo "✅ Setup complete inside speculative-beam-decoding/"

