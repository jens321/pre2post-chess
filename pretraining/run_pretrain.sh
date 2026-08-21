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
#
# No venv activation needed: the run goes through `uv run`, which syncs the
# environment from uv.lock first. Override the binary with UV=/path/to/uv.
#
# If `cluster-env.sh` sits next to this script it is sourced first; that is
# where the environment layout lives (e.g. a venv on node-local disk instead of
# a network filesystem). It is tracked, so a fresh clone gets the fast path; the
# `-f` guard keeps this working if someone deletes it.

set -euo pipefail

# Directory holding this script, i.e. the `pretraining/` subtree -- not the repo
# root. Also the uv project root (pyproject.toml / uv.lock live here).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=/dev/null
[[ -f "${SCRIPT_DIR}/cluster-env.sh" ]] && source "${SCRIPT_DIR}/cluster-env.sh"

UV="${UV:-$(command -v uv || true)}"
if [[ -z "${UV}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV}" ]]; then
  echo "ERROR: uv not found on PATH. Install it or set UV=/path/to/uv." >&2
  exit 1
fi

# ---- Configurable inputs (override via env) ---------------------------------
# Either CONFIG (single yaml) or CONFIG_DIR (run every yaml in the dir).
CONFIG="${CONFIG:-}"
CONFIG_DIR="${CONFIG_DIR:-}"
# Directory of tokenized pretraining shards (.npy). Overrides data.txt_path.
# Leave empty to use whatever txt_path the config already carries.
DATA_DIR="${DATA_DIR:-}"
# Where checkpoints are written. Maps to training.save_dir.
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints}"
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
# Optional space-separated OmegaConf overrides.
OVERRIDES="${OVERRIDES:-}"
# Mixed precision for the accelerate launcher: bf16, fp16, or no.
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
# Rendezvous port for multi-GPU runs. Defaults to a free port so that concurrent
# runs on one node do not collide. Must be a real port: 0 makes non-zero ranks
# hang forever trying to connect to MASTER_PORT=0.
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$(
  python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' \
    2>/dev/null || echo 29500
)}"

if [[ -z "${CONFIG}" && -z "${CONFIG_DIR}" ]]; then
  echo "ERROR: set CONFIG=<file.yaml> or CONFIG_DIR=<dir of yamls>." >&2
  exit 1
fi

# A typo'd DATA_DIR should fail here, not a few minutes into the first run.
if [[ -n "${DATA_DIR}" && ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: DATA_DIR='${DATA_DIR}' is not a directory." >&2
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
echo "[pretrain] data_dir=${DATA_DIR:-<from config data.txt_path>}"
echo "[pretrain] output_dir=${OUTPUT_DIR}"
echo "[pretrain] mixed_precision=${MIXED_PRECISION} | main_process_port=${MAIN_PROCESS_PORT}"
echo "[pretrain] venv=${UV_PROJECT_ENVIRONMENT:-${SCRIPT_DIR}/.venv}"
echo "=========================================="

for cfg in "${configs[@]}"; do
  echo "------------------------------------------"
  echo "[pretrain] config: ${cfg}"
  echo "------------------------------------------"

  extra_args=()
  [[ -n "${DATA_DIR}" ]] && extra_args+=(--data_dir "${DATA_DIR}")
  [[ -n "${TEST_DATA_DIR}" ]] && extra_args+=(--test_data_dir "${TEST_DATA_DIR}")
  override_args=()
  if [[ -n "${OVERRIDES}" ]]; then
    read -r -a overrides <<< "${OVERRIDES}"
    override_args+=(--override "${overrides[@]}")
  fi
  launch_args=(
    --num_processes "${NUM_GPUS}"
    --num_machines 1
    --mixed_precision "${MIXED_PRECISION}"
    --dynamo_backend no
    --main_process_port "${MAIN_PROCESS_PORT}"
  )
  if (( NUM_GPUS > 1 )); then
    launch_args+=(--multi_gpu)
  fi

  "${UV}" run --project "${SCRIPT_DIR}" \
    accelerate launch "${launch_args[@]}" \
    "${SCRIPT_DIR}/scripts/train/train_hf.py" \
    --config "${cfg}" \
    --auto_resume \
    --output_dir "${OUTPUT_DIR}" \
    --max_checkpoints "${MAX_CHECKPOINTS}" \
    "${override_args[@]}" \
    "${extra_args[@]}"
done

echo "[pretrain] all runs complete."
