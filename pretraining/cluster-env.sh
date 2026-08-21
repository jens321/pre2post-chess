# Environment layout for this project. Sourced by run_pretrain.sh.
#
# /data is NFS with lookupcache=none, so every path component of every lookup is
# a network round trip. A venv there costs 69 s per `uv run` launch instead of
# 1.8 s -- hence the environment goes node-local and only the lockfile stays on
# /data. /tmp is wiped per pod; `uv run` re-syncs from uv.lock automatically.
#
# Tracked in git on purpose: if it were ignored, a fresh clone would fall back to
# a .venv on /data and just be slow, with no error.

# Machine scope, shared with every uv project here. ~/.bashrc sources the same
# file, but `bash run_pretrain.sh` never reads .bashrc, so this line is what
# covers batch jobs. Both it and ~/.config/uv/uv.toml (uv's cache-dir) hang off
# $HOME, and containers start as root: a job spec that does not export
# HOME=/data/home/$USER finds neither and puts the interpreter and cache on the
# PVC. The fallback keeps such a job fast anyway and logs which path it took.
# shellcheck source=/dev/null
if [ -f "$HOME/.config/bonete-env.sh" ]; then
  . "$HOME/.config/bonete-env.sh"
else
  echo "[cluster-env] warning: \$HOME/.config/bonete-env.sh not found (HOME=${HOME:-<unset>})." >&2
  echo "[cluster-env] Using inline defaults. Export HOME=/data/home/\$USER in the job spec to use the shared config." >&2
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/tmp/uv-python}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
fi

# Project x machine scope: must NOT be a global export, or every uv project on
# the box resolves to the same venv. Also mirrored in .vscode/launch.json and
# .vscode/tasks.json (VS Code cannot source a shell file) -- change all three.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/uv-envs/pre2post-pretraining}"

# HF_HOME/TORCH_HOME deliberately unset: a few large files (790 MB across 19),
# which this mount serves fine. Moving them re-downloads every pod for no win.
