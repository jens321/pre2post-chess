#!/bin/bash
# Launch chess pretraining (single machine, via accelerate).
#
# Runs `accelerate launch scripts/train/train_hf.py` either on a single config
# file or on every *.yaml in a config directory (sequentially).
#
# Usage:
#   # one config
#   CONFIG=config/configs/6p5e18/410m_alpha1.000.yaml \
#   DATA_DIR=/path/to/tokenized_shards \
#   NUM_GPUS=8 WANDB_ENTITY=my-team \
#   bash run_pretrain.sh
#
#   # a whole sweep directory (one run after another)
#   CONFIG_DIR=config/configs/6p5e18 \
#   DATA_DIR=/path/to/tokenized_shards \
#   NUM_GPUS=8 \
#   bash run_pretrain.sh
#
# All inputs are environment variables (with sensible defaults below).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable inputs (override via env) ---------------------------------
# Either CONFIG (single yaml) or CONFIG_DIR (run every yaml in the dir).
CONFIG="${CONFIG:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
# Directory of tokenized pretraining shards (.npy). Maps to data.txt_path.
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/pretrain_v1_54b}"
# Where checkpoints are written. Maps to training.save_dir.
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints}"
# Optional base path for held-out eval rollout test files (data.test_data_dir).
TEST_DATA_DIR="${TEST_DATA_DIR:-}"
# GPUs for this run.
NUM_GPUS="${NUM_GPUS:-1}"
# Old checkpoints to keep on disk (0 = unlimited).
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-3}"
# Weights & Biases team/entity. Leave empty to use your default wandb account.
WANDB_ENTITY="${WANDB_ENTITY:-}"
# Weights & Biases project. The configs carry no logging block, so set it here.
WANDB_PROJECT="${WANDB_PROJECT:-chess-pretraining}"

if [[ -z "${CONFIG}" && -z "${CONFIG_DIR}" ]]; then
  echo "ERROR: set CONFIG=<file.yaml> or CONFIG_DIR=<dir of yamls>." >&2
  exit 1
fi

# Build the list of configs to run.
configs=()
if [[ -n "${CONFIG}" ]]; then
  configs+=("${CONFIG}")
fi
if [[ -n "${CONFIG_DIR}" ]]; then
  while IFS= read -r f; do configs+=("${f}"); done < <(find "${CONFIG_DIR}" -maxdepth 1 -name '*.yaml' | sort)
fi

# Single-GPU performance / determinism knobs.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
[[ -n "${WANDB_ENTITY}" ]] && export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT}"

echo "=========================================="
echo "[pretrain] ${#configs[@]} config(s) | ${NUM_GPUS} GPU(s)"
echo "[pretrain] data_dir=${DATA_DIR}"
echo "[pretrain] output_dir=${OUTPUT_DIR}"
echo "=========================================="

for cfg in "${configs[@]}"; do
  echo "------------------------------------------"
  echo "[pretrain] config: ${cfg}"
  echo "------------------------------------------"

  extra_args=()
  [[ -n "${TEST_DATA_DIR}" ]] && extra_args+=(--test_data_dir "${TEST_DATA_DIR}")

  accelerate launch --num_processes "${NUM_GPUS}" \
    "${REPO_ROOT}/scripts/train/train_hf.py" \
    --config "${cfg}" \
    --auto_resume \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_checkpoints "${MAX_CHECKPOINTS}" \
    "${extra_args[@]}"
done

echo "[pretrain] all runs complete."
