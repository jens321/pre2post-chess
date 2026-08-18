# Chess pretraining

## Environment

This directory is a self-contained [uv](https://docs.astral.sh/uv/) project
(`pyproject.toml` + `uv.lock`). Create the environment with:

```bash
cd pretraining
uv sync            # creates pretraining/.venv from uv.lock
```

You do **not** need to activate anything — prefix commands with `uv run`, which
re-syncs the venv before each invocation:

```bash
uv run python scripts/train/train_hf.py --help
```

`run_pretrain.sh` already wraps its `accelerate launch` in `uv run`, so it works
from any shell as long as `uv` is on `PATH` (or `UV=/path/to/uv` is set).

Core dependencies: `torch`, `accelerate`, `transformers`, `omegaconf`,
`python-chess`, `numpy`, `pandas`, `pyarrow`, `scipy`, `wandb`. Versions are
pinned in `pyproject.toml`; change them there and re-run `uv lock`.
Multi-GPU training is driven by `accelerate launch`.

`environment.yml` is the older conda recipe, kept only for reference — prefer uv.

### `${REPO_ROOT}` placeholder

A few **default** path strings inside the Python files (eval-data and stockfish
defaults in `evaluation/` and `training/trainer_hf.py`) contain the literal
`${REPO_ROOT}` — these are placeholders not used in the normal workflow, because
real paths are supplied via the config YAML and CLI flags. Override them there
rather than editing the defaults.

## Data

Training reads a directory of **tokenized `.npy` shards** (`data.txt_path`, or
`--data_dir`). Point `DATA_DIR` at your tokenized shard directory. If you leave
`DATA_DIR` unset, `run_pretrain.sh` omits `--data_dir` and the config's own
`data.txt_path` is used as-is (the shipped configs point at a **placeholder**
`/data/pretrain_v1_54b`).

Validation shards are chosen in this order:

1. `data.eval_txt_files` — explicit paths (dirs, globs, or single files). The
   shipped configs list **placeholder** paths under `/data/eval/`; replace them
   with your own.
2. If those match nothing, the trainer warns and falls back to holding out the
   last `data.eval_holdout` shards of the training set (default `1`).

## Logging

The configs carry no `logging:` block. W&B project and entity come from the
`WANDB_PROJECT` and `WANDB_ENTITY` environment variables (project defaults to
`chess-pretraining`); the run name is `training.experiment_name`. Set
`WANDB_MODE=offline` or `disabled` to change or turn off syncing. To pin these
per config, add an optional `logging:` block with `project:` / `entity:` /
`notes:` / `tags:` — it takes precedence over the environment.

---

## Quick start

```bash
cd pretraining   # run_pretrain.sh handles the venv via `uv run`

# One config:
CONFIG=config/configs/6p5e18/410m_alpha1.000.yaml \
DATA_DIR=/path/to/tokenized_shards \
OUTPUT_DIR=/path/to/checkpoints \
NUM_GPUS=8 \
WANDB_ENTITY=my-team \
bash run_pretrain.sh

# A whole sweep directory (runs each yaml in turn):
CONFIG_DIR=config/configs/6p5e18 \
DATA_DIR=/path/to/tokenized_shards \
NUM_GPUS=8 \
bash run_pretrain.sh
```

`run_pretrain.sh` calls `accelerate launch scripts/train/train_hf.py` with
`--auto_resume`, so re-running picks up from the latest checkpoint. All knobs are
environment variables: `CONFIG` / `CONFIG_DIR`, `DATA_DIR`, `OUTPUT_DIR`,
`TEST_DATA_DIR`, `NUM_GPUS`, `MAX_CHECKPOINTS`, `WANDB_ENTITY`, `WANDB_PROJECT`.

---

## Configs

Each YAML is named `{size}_alpha{α}.yaml`. **`α` is the fraction of the total
compute budget `C_total` allocated to pretraining** — pretrain token count scales
linearly with `α` (e.g. for 410m at `C_total=6.5e18`: `α=1.0 → 2.64B` tokens,
`α=0.05 → 0.13B`). The remaining compute is reserved for the post-training
(SFT/RL) stages. Override any field on the CLI, e.g.
`--override training.batch_size=64`.

### Model configs

All models are Qwen3-style decoder-only transformers with GQA
(`num_key_value_heads=4`, `head_dim=128`), `block_size=1024`, and no dropout.

| Size  | Layers | d_model | Heads | KV heads | FFN (intermediate) |
|-------|:------:|:-------:|:-----:|:--------:|:------------------:|
| 20m   | 6      | 512     | 4     | 4        | 1536               |
| 32m   | 8      | 512     | 8     | 4        | 1536               |
| 50m   | 12     | 512     | 8     | 4        | 1536               |
| 100m  | 12     | 768     | 12    | 4        | 2304               |
| 200m  | 24     | 768     | 12    | 4        | 2304               |
| 410m  | 28     | 1024    | 16    | 4        | 3072               |
| 680m  | 30     | 1280    | 20    | 4        | 3840               |
| 1000m | 32     | 1536    | 24    | 4        | 4608               |

### Training configs

Shared across all sizes (override per run via `--override`):

| Setting                  | Value                                  |
|--------------------------|----------------------------------------|
| Optimizer                | AdamW, betas `(0.9, 0.95)`, wd `0.1`   |
| Learning rate            | `1e-3`                                  |
| LR schedule              | cosine, `eta_min=1e-4`, warmup `5%`    |
| Per-device batch size    | `32`                                    |
| Gradient accumulation    | `2`                                     |
| Effective batch (tokens) | `32 × NUM_GPUS × 2 × 1024`             |
| Max grad norm            | `1.0`                                   |
| Epochs                   | `1` (token budget set by `α`)          |
| Sequence length          | `1024`                                  |
| Precision                | bf16 (via `accelerate`)                 |

### Device configs

| Setting        | Value / control                | Set via                    |
|----------------|--------------------------------|----------------------------|
| GPUs per run   | `NUM_GPUS` (default 1)         | `run_pretrain.sh` env var  |
| Multi-GPU      | DDP via `accelerate launch`    | automatic                  |
| Precision      | bf16                           | `accelerate`               |
| Sweep over configs | one run after another      | `CONFIG_DIR`               |

The effective batch size scales with `NUM_GPUS` (`32 × NUM_GPUS × 2 × 1024`
tokens); to match a different GPU count, adjust
`training.gradient_accumulation_steps` (via `--override`) accordingly.
