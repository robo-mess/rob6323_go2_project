#!/usr/bin/env bash
set -euo pipefail

# Quote args safely for SSH -> remote shell
ARGS="$(printf '%q ' "$@")"

ssh -o StrictHostKeyChecking=accept-new burst \
  "cd ~/rob6323_go2_project && sbatch \
    --job-name=rob6323_\$USER \
    --mail-user=\$USER@nyu.edu \
    train_s.slurm ${ARGS}"
