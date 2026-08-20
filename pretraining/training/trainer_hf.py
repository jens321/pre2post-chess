import json
import pathlib, shutil
import os, sys, argparse, pathlib
import time
from tqdm import tqdm

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from accelerate import Accelerator
from .data_utils import create_dataloader
from transformers import AutoModelForCausalLM, AutoConfig
from llm_tokens.chess.tokenizer_factory import init_tokenizer
from .optim_sched import build_optimizer, build_scheduler
import torch
from evaluation.example_evaluator import HumanGamesEvaluator, PuzzlesEvaluator, RandomGamesEvaluator

class HFTrainer:
    def __init__(self, cfg, run_config_path=None):
        self.cfg = cfg
        self.run_cfg_path = run_config_path
        self._hf_upload_thread = None   # track background upload thread
        self._hf_repo_cleared = False   # clear repo once per training run
        backend = self._logging_cfg().get("backend", "wandb")
        # Enable mixed precision training for better GPU utilization
        mixed_precision = cfg.training.get("mixed_precision", "no")  # bf16, fp16, or no
        gradient_accumulation_steps = cfg.training.get("gradient_accumulation_steps", 1)
        self.acc = Accelerator(
            log_with=("wandb" if backend == "wandb" else backend),
            mixed_precision=mixed_precision if mixed_precision != "no" else None,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        self._init_all()

    def _logging_cfg(self):
        """Tracker settings from the optional `logging:` config block.

        Configs ship without a `logging:` block: entity/project come from the
        WANDB_ENTITY / WANDB_PROJECT environment variables instead.
        """
        return self.cfg.get("logging", None) or {}

    # ---------------- core builders ----------------
    def _init_all(self):
        # Note: torch is imported at module level (line 13)
        # Don't use local 'import torch' in this function as it causes UnboundLocalError
        cfg = self.cfg
        self.mcfg, self.tcfg, self.dcfg, self.tokcfg = cfg.model, cfg.training, cfg.data, cfg.tokenizer

        # ----- tokenizer -----
        self.tok = init_tokenizer(
            name=self.tokcfg.name,
            config=self.tokcfg
        )

        # ----- dataloaders -----
        all_files = self._expand_txt_files(self.dcfg)

        # Explicit eval shards win; otherwise hold out the last `eval_holdout`
        # training shards. Configs ship placeholder eval paths, so a set of
        # eval_txt_files that resolves to nothing warns and falls back rather
        # than aborting the run.
        self.eval_files = []
        eval_txt_files = self.dcfg.get("eval_txt_files", None)
        if eval_txt_files:
            self.eval_files = self._expand_paths(eval_txt_files)
            if not self.eval_files:
                self.acc.print(f"[warn] data.eval_txt_files matched no files ({list(eval_txt_files)}); "
                               f"falling back to data.eval_holdout")

        if self.eval_files:
            self.train_files = [p for p in all_files if p not in set(self.eval_files)]
        else:
            k = int(self.dcfg.get("eval_holdout", 1))
            k = max(0, min(k, len(all_files) - 1))
            self.train_files = all_files[:-k] if k > 0 else all_files
            self.eval_files = all_files[-k:] if k > 0 else []

        # Deterministic seed for shard shuffling (reproducible across resume)
        self._data_seed = int(self.tcfg.get("seed", 42))

        # Optimize data loading for better GPU utilization
        num_workers = int(self.tcfg.get("num_workers", 4))  # Default to 4 workers
        prefetch_factor = self.tcfg.get("prefetch_factor", 2 if num_workers > 0 else None)
        persistent_workers = self.tcfg.get("persistent_workers", True if num_workers > 0 else False)

        self._cache_size = int(self.tcfg.get("cache_size", 0))  # 0 = auto-detect full shard
        self.train_loader = create_dataloader(
                txt_files=self.train_files,
                tokenizer=self.tok,
                batch_size=self.tcfg.batch_size,
                seq_len=self.mcfg.block_size,
                num_workers=num_workers,
                shuffle=False,
                cache_size=self._cache_size,
                dataset_shuffle=True,
                num_shards=self.dcfg.get("num_shards", None),
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
                seed=self._data_seed,
            )
        
        # Store these for later use when recreating dataloader
        self.dataloader_kwargs = {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
        }

        self.eval_loader = None
        if self.eval_files:
            self.eval_loader = create_dataloader(
            txt_files=self.eval_files,
            tokenizer=self.tok,
            batch_size=self.tcfg.get("eval_batch_size", self.tcfg.batch_size),
            seq_len=self.mcfg.block_size,
            num_workers=0,
            shuffle=False,
        )

        # vocab size (tolerate both interfaces)
        if hasattr(self.tok, "get_vocab_size"):
            self.vocab_size = int(self.tok.get_vocab_size())
            print("Vocab size: ", self.vocab_size)
        else:
            self.vocab_size = int(len(self.tok.get_vocab()))

        # ----- model -----
        # Support three modes:
        # 1. pretrained_model + init_from_scratch=False: Load pretrained weights
        # 2. pretrained_model + init_from_scratch=True: Use architecture but random init all weights
        # 3. No pretrained_model: Build from config (architecture must be specified)
        pretrained_model = self.mcfg.get("pretrained_model", None)
        init_from_scratch = self.mcfg.get("init_from_scratch", False)  # Use architecture but random weights
        reinit_embeddings = self.mcfg.get("reinit_embeddings", True)  # Whether to reinit embeddings after resize
        
        if pretrained_model and not init_from_scratch:
            # Mode 1: Load pretrained weights from HuggingFace
            self.acc.print(f"Loading pretrained model: {pretrained_model}")
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.tcfg.get("mixed_precision") == "bf16" else torch.float32,
            )
            
            # Resize token embeddings if vocab size differs
            old_vocab_size = self.model.config.vocab_size
            if old_vocab_size != self.vocab_size:
                self.acc.print(f"Resizing token embeddings from {old_vocab_size} to {self.vocab_size}")
                self.model.resize_token_embeddings(self.vocab_size)
                self.model.config.vocab_size = self.vocab_size
                
                if reinit_embeddings:
                    self._reinit_embeddings()
            
            self.acc.print(f"[model] Loaded {pretrained_model} with pretrained weights")
            
        elif pretrained_model and init_from_scratch:
            # Mode 2: Use architecture from pretrained_model but initialize ALL weights randomly
            self.acc.print(f"Using architecture from {pretrained_model} but initializing from scratch")
            
            # Load config only (not weights)
            hf_config = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
            
            # Override vocab size
            hf_config.vocab_size = self.vocab_size
            hf_config.bos_token_id = self.tok.bos_id()
            hf_config.eos_token_id = self.tok.eos_id()
            hf_config.pad_token_id = self.tok.pad_id()

            # Optionally override block_size/max_position_embeddings
            if self.mcfg.get("block_size"):
                if hasattr(hf_config, 'max_position_embeddings'):
                    hf_config.max_position_embeddings = self.mcfg.block_size
                if hasattr(hf_config, 'n_positions'):
                    hf_config.n_positions = self.mcfg.block_size
            
            # Create model from config (random initialization)
            self.model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
            # self.model.resize_token_embeddings(self.vocab_size)
            
            self.acc.print(f"[model] Initialized {pretrained_model} architecture from scratch "
                          f"(hidden_size={hf_config.hidden_size if hasattr(hf_config, 'hidden_size') else hf_config.n_embd}, "
                          f"layers={hf_config.num_hidden_layers if hasattr(hf_config, 'num_hidden_layers') else hf_config.n_layer})")
            
        else:
            # Mode 3: Build model from scratch using Qwen3 config directly
            self.acc.print(f"Initializing Qwen3 model from scratch")

            from transformers import Qwen3Config
            hf_config = Qwen3Config(
                # Fixed Qwen3 template values
                attention_bias=False,
                hidden_act="silu",
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                rope_scaling=None,
                rope_theta=1000000,
                sliding_window=None,
                tie_word_embeddings=True,
                torch_dtype="bfloat16",
                use_cache=True,
                use_sliding_window=False,
                # Per-model values from config
                vocab_size=self.vocab_size,
                hidden_size=self.mcfg.n_embed,
                intermediate_size=self.mcfg.intermediate_size,
                num_hidden_layers=self.mcfg.n_layer,
                num_attention_heads=self.mcfg.n_head,
                num_key_value_heads=self.mcfg.get("num_key_value_heads", 4),
                head_dim=self.mcfg.get("head_dim", 128),
                max_position_embeddings=self.mcfg.get("block_size", 2048),
                max_window_layers=self.mcfg.n_layer,
                attention_dropout=self.mcfg.get("dropout", 0.0),
            )
            
            self.model = AutoModelForCausalLM.from_config(hf_config, trust_remote_code=True)
            self.acc.print(f"[model] Created Qwen3 model from scratch")
        self.acc.print(f"[model] Model size: {sum(p.numel() for p in self.model.parameters())}")   
        # Load pretrained weights if specified (before accelerator.prepare)
        if "pretrained_weights" in self.tcfg and self.tcfg.pretrained_weights:
            self._load_pretrained_weights(self.tcfg.pretrained_weights)
        
        # Optional: Compile model for additional speedup (PyTorch 2.0+)
        if self.tcfg.get("compile_model", False):
            compile_mode = self.tcfg.get("compile_mode", "reduce-overhead")
            self.acc.print(f"[optimization] Compiling model with mode={compile_mode}")
            try:
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode=compile_mode)
                    self.acc.print("[optimization] Model compiled successfully (first iteration will be slower)")
                else:
                    self.acc.print("[warn] torch.compile not available, skipping compilation")
            except Exception as e:
                self.acc.print(f"[warn] Model compilation failed: {e}")
        
        print("="*50)
        print("Model: ", self.model)
        print("="*50)

        # ----- optimizer & scheduler -----
        self.optimizer = build_optimizer(self.model, {
            "name": self.tcfg.optimizer.name,
            "lr": self.tcfg.optimizer.lr,
            "weight_decay": self.tcfg.optimizer.get("weight_decay", 0.1),
            "betas": tuple(self.tcfg.optimizer.get("betas", (0.9, 0.95))),
            "exclude_bias_and_norm": True,
        })

        self.epochs = self.tcfg.epochs
        steps_per_epoch = len(self.train_loader)

        # Compute total optimizer steps for the scheduler.
        # If pretrain_tokens is set, use that to determine total steps (isoflop).
        # Otherwise fall back to dataloader length.
        import math
        pretrain_tokens = self.dcfg.get("pretrain_tokens", None)
        gradient_accumulation_steps = self.acc.gradient_accumulation_steps
        if pretrain_tokens is not None:
            # With DDP, total tokens per opt step = bs * seq * ga * num_gpus
            tokens_per_opt_step = (self.tcfg.batch_size * self.mcfg.block_size
                                   * gradient_accumulation_steps * self.acc.num_processes)
            self._max_steps = int(pretrain_tokens) // tokens_per_opt_step
            # Scheduler total = max_steps * num_processes (pre-DDP units for Accelerate compensation)
            scheduler_total_steps = self._max_steps * self.acc.num_processes
        else:
            self._max_steps = None
            scheduler_total_steps = max(1, math.ceil(steps_per_epoch / gradient_accumulation_steps)) * self.tcfg.epochs

        warmup_ratio = self.tcfg.scheduler.get("warmup_ratio", 0.05)
        warmup_steps = self.tcfg.scheduler.get("warmup_steps", None)
        if warmup_steps is None:
            warmup_steps = int(scheduler_total_steps * warmup_ratio)

        self.acc.print(f"[scheduler] scheduler_total={scheduler_total_steps}, warmup_steps={warmup_steps}, "
                       f"max_steps={self._max_steps}, num_processes={self.acc.num_processes}")

        self.scheduler = build_scheduler(
            self.optimizer,
            {
                "name": self.tcfg.scheduler.name,
                "eta_min": self.tcfg.scheduler.get("eta_min", 0.0),
                "warmup_steps": warmup_steps,
            },
            total_steps=scheduler_total_steps,
        )

        # ----- accelerator prep -----
        # Note: Accelerate's AcceleratedScheduler compensates for DDP automatically
        # (steps the underlying scheduler num_processes times per .step() call),
        # so scheduler should be built with the pre-prepare dataloader length.
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
            self.eval_loader,
        ) = self.acc.prepare(self.model, self.optimizer, self.train_loader, self.scheduler, self.eval_loader)

        self.acc.print(f"[accelerate] Using device: {self.acc.device}")
        self.acc.print(f"[train] pre-prepare loader len={steps_per_epoch}, "
                       f"post-prepare loader len={len(self.train_loader)}, "
                       f"num_processes={self.acc.num_processes}, "
                       f"grad_accum={self.acc.gradient_accumulation_steps}")

        # ----- dirs & tracking -----
        run_name = self._auto_run_name(cfg)
        exp_name = self.tcfg.get("experiment_name", None)
        if exp_name:
            self.save_dir = pathlib.Path(self.tcfg.get("save_dir", "checkpoints")) / exp_name
        else:
            self.save_dir = pathlib.Path(self.tcfg.get("save_dir", "checkpoints/gpt")) / run_name
        self._save_interval_cfg = self.tcfg.get("save_interval", None)
        self.save_interval = self._save_interval_cfg or 1000  # will be recalculated in train()
        self.log_interval = self.tcfg.get("log_interval", 50)
        self.resume_path = self.tcfg.get("resume", None)
        self._local_log_path = self.save_dir / "metrics.jsonl"
        self.smoke_test = bool(self.tcfg.get("smoke_test", False))
        
        # Training state for resuming
        self.current_step = 0
        self.current_epoch = 0

        if self.acc.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # init trackers on ALL ranks (Accelerate will manage rank-0 setup)
        project_name = (self._logging_cfg().get("project", None)
                        or os.environ.get("WANDB_PROJECT", "chess-pretraining"))
        self.acc.print(f"Initing trackers: project={project_name} run={run_name}")
        self.acc.init_trackers(
            project_name,                      
            init_kwargs=self._tracker_init_kwargs(run_name, self._tracker_cfg())
        )
        if self.acc.is_main_process:
            import wandb
            if wandb.run is not None:
                wandb.run.name = run_name
                self.acc.print(f"[wandb] effective name: {wandb.run.name} (id={wandb.run.id})")

        if self.acc.is_main_process:
            self._snapshot_config()

        # Auto-resume: find latest checkpoint if enabled and no explicit resume path
        auto_resume = bool(self.tcfg.get("auto_resume", False))
        if not self.resume_path and auto_resume:
            self.resume_path = self._find_latest_checkpoint()
            if self.resume_path:
                self.acc.print(f"[auto_resume] Found latest checkpoint: {self.resume_path}")

        if self.resume_path:
            self.acc.load_state(self.resume_path)
            self._load_training_state(self.resume_path)
            self.acc.print(f"[train] resumed from {self.resume_path} at step {self.current_step}, epoch {self.current_epoch}")

        # MFU tracking state
        self._last_log_time = time.time()
        self._last_log_step = 0
        self._tokens_per_batch = self.tcfg.batch_size * self.mcfg.block_size
        self._flops_per_token = self._estimate_flops_per_token()
        self._gpu_peak_tflops = self.tcfg.get("gpu_peak_tflops", 312)  # A100 bf16 default

        if self.smoke_test and self.acc.is_main_process:
            xb, yb = next(iter(self.train_loader))
            self.acc.print(f"[smoke] x={tuple(xb.shape)} y={tuple(yb.shape)}")

    def _reinit_embeddings(self):
        """Reinitialize all token embeddings (for when vocab changes significantly)."""
        self.acc.print(f"[fix] Reinitializing all token embeddings")
        with torch.no_grad():
            embed_layer = self.model.get_input_embeddings()
            lm_head = self.model.get_output_embeddings()
            
            embed_dim = embed_layer.weight.shape[1]
            std = (1.0 / embed_dim) ** 0.5
            
            torch.nn.init.normal_(embed_layer.weight, mean=0.0, std=std)
            
            if lm_head is not None:
                torch.nn.init.normal_(lm_head.weight, mean=0.0, std=std)
                if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                    torch.nn.init.zeros_(lm_head.bias)
        
        self.acc.print(f"[fix] Reinitialized {self.vocab_size} token embeddings")
    
    def _load_pretrained_weights(self, path):
        """Load only model weights from a checkpoint (not optimizer/scheduler)."""
        import torch
        from pathlib import Path
        path = Path(path)
        
        if path.is_dir():
            # Accelerate checkpoint directory structure
            model_path = path / "pytorch_model.bin"
            if not model_path.exists():
                # Try alternative names
                alt_paths = list(path.glob("*.bin")) + list(path.glob("*.pt")) + list(path.glob("*.pth"))
                if alt_paths:
                    model_path = alt_paths[0]
                else:
                    raise FileNotFoundError(f"No model weights found in {path}")
        else:
            model_path = path
        
        self.acc.print(f"[init] Loading pretrained weights from {model_path}")
        state_dict = torch.load(model_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Load weights with strict=False to allow partial loading
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            self.acc.print(f"[init] Missing keys: {missing}")
        if unexpected:
            self.acc.print(f"[init] Unexpected keys: {unexpected}")
        
        self.acc.print(f"[init] Successfully loaded pretrained weights")
    
    def _save_training_state(self, checkpoint_dir):
        """Save training state for proper resuming."""
        import json
        if self.acc.is_main_process:
            grad_accum = self.acc.gradient_accumulation_steps
            state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
            state = {
                "step": self.current_step,
                "epoch": self.current_epoch,
                "total_batches": self.current_step * grad_accum,
                "grad_accum_steps": grad_accum,
                "data_seed": self._data_seed,
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

    def _load_training_state(self, checkpoint_dir):
        """Load training state when resuming."""
        import json
        state_file = pathlib.Path(checkpoint_dir) / "training_state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            self.current_step = state.get("step", 0)
            self.current_epoch = state.get("epoch", 0)
            self._data_seed = state.get("data_seed", self._data_seed)
        else:
            self.acc.print(f"[warn] No training_state.json found in {checkpoint_dir}, starting from step 0")

    def _find_latest_checkpoint(self):
        """Find the resume checkpoint in save_dir for auto-resume."""
        import re
        if not self.save_dir.exists():
            return None
        # New format: single latest/ directory
        latest_dir = self.save_dir / "latest"
        if latest_dir.is_dir() and (latest_dir / "training_state.json").exists():
            return str(latest_dir)
        # Legacy format: find highest step* directory
        ckpt_dirs = sorted(
            (d for d in self.save_dir.iterdir()
             if d.is_dir() and re.match(r"step\d+$", d.name)),
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )
        if ckpt_dirs and (ckpt_dirs[-1] / "training_state.json").exists():
            return str(ckpt_dirs[-1])
        return None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent max_checkpoints."""
        import re
        max_ckpts = int(self.tcfg.get("max_checkpoints", 0))
        if max_ckpts <= 0:
            return
        ckpt_dirs = sorted(
            (d for d in self.save_dir.iterdir()
             if d.is_dir() and re.match(r"step\d+$", d.name)),
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )
        while len(ckpt_dirs) > max_ckpts:
            old = ckpt_dirs.pop(0)
            self.acc.print(f"[cleanup] Removing old checkpoint: {old}")
            shutil.rmtree(old)

    # ---------------- flops / mfu ----------------
    def _estimate_flops_per_token(self):
        """Estimate FLOPs per token for forward + backward pass (approx 6N)."""
        base_model = self.acc.unwrap_model(self.model) if hasattr(self.acc, 'unwrap_model') else self.model
        n_params = sum(p.numel() for p in base_model.parameters())
        return 6 * n_params

    @staticmethod
    def _board_from_pgn(state_text: str):
        """Parse a PGN movetext string into a chess.Board."""
        import chess, chess.pgn, io
        state_text = state_text.strip()
        try:
            return chess.Board(state_text)  # try FEN
        except Exception:
            pass
        game = chess.pgn.read_game(io.StringIO(state_text))
        return game.end().board() if game else chess.Board()

    # ---------------- utils ----------------
    def _auto_run_name(self, cfg):
        """Automatically build run name from model, optimizer, and data configs."""
        exp_name = cfg.training.get("experiment_name", None)
        if exp_name:
            explicit_run_name = cfg.training.get("run_name", None)
            return explicit_run_name or exp_name

        explicit_run_name = cfg.training.get("run_name", None)
        if explicit_run_name:
            return explicit_run_name

        m = cfg.model
        t = cfg.training
        d = cfg.data
        tok = cfg.tokenizer
        stem = "_shards"

        pretrained = m.get("pretrained_model", None)
        if pretrained:
            model_name = pretrained.split("/")[-1].replace("-", "_")
        else:
            arch = m.get("architecture", "gpt2")
            model_name = f"scratch_{arch}"

        # ---- robust model shape for naming ----
        # Prefer explicit cfg fields if present; otherwise derive from HF config.
        mc = getattr(self, "model", None)
        hc = getattr(mc, "config", None) if mc is not None else None

        def pick(cfg_key, *hf_keys, default="NA"):
            v = m.get(cfg_key, None)
            if v is not None:
                return v
            if hc is not None:
                for k in hf_keys:
                    if hasattr(hc, k) and getattr(hc, k) is not None:
                        return getattr(hc, k)
            return default

        n_layer = pick("n_layer", "num_hidden_layers", "n_layer")
        n_head  = pick("n_head", "num_attention_heads", "n_head")
        n_embed = pick("n_embed", "hidden_size", "n_embd")
        num_shards = d.get("num_shards", "NA")

        return (
            f"hf_{model_name}_{tok.name}_data{num_shards}_ctx{self.mcfg.block_size}"
            f"_L{n_layer}H{n_head}E{n_embed}"
            f"_bs{t.batch_size}_lr{t.optimizer.lr}_{stem}"
        )

    def _tracker_cfg(self):
        c = self.cfg
        return {
            "epochs": c.training.epochs,
            "batch_size": c.training.batch_size,
            "lr": c.training.optimizer.lr,
            "seq_len": c.model.block_size,
            "vocab_size": self.vocab_size,
            "tokenizer": c.tokenizer.name,
            "pretrained_model": c.model.get("pretrained_model", None),
        }

    def _tracker_init_kwargs(self, run_name: str, cfg: dict):
        # accept both "wandb" and ["wandb"]
        kinds = self.acc.log_with if isinstance(self.acc.log_with, (list, tuple)) else [self.acc.log_with]
        if "wandb" not in kinds:
            return {}
        lg = self._logging_cfg()
        return {
            "wandb": {
                # project is positional in init_trackers; don't duplicate here
                "entity": lg.get("entity", None) or os.environ.get("WANDB_ENTITY", None),
                "name": run_name,                 
                "notes": lg.get("notes", None),
                "tags": lg.get("tags", []),
                "config": cfg,
            }
        }

    def _snapshot_config(self):
        if not self.run_cfg_path:
            return
        dst = self.save_dir / "config.yaml"
        try:
            shutil.copy(self.run_cfg_path, dst)
        except Exception as e:
            self.acc.print(f"[warn] config snapshot failed: {e}")

    # ---------------- train loop ----------------
    def train(self):
        self.model.train()
        steps_per_epoch = len(self.train_loader)
        resume_skip_done = False  # Flag to ensure we only skip once on resume
        gradient_accumulation_steps = self.tcfg.get("gradient_accumulation_steps", 1)

        # max_steps computed in _init_all from pretrain_tokens
        max_steps = self._max_steps

        # Compute total optimizer steps for ratio-based intervals
        opt_steps_per_epoch = max(1, steps_per_epoch // gradient_accumulation_steps)
        total_opt_steps = max_steps if max_steps is not None else (opt_steps_per_epoch * self.epochs)

        # All intervals: use absolute value from config if set, otherwise compute from ratio
        eval_interval = self.tcfg.get("eval_interval", None)
        if eval_interval is None:
            eval_interval = max(10, int(total_opt_steps * self.tcfg.get("eval_ratio", 0.1)))

        save_hf_interval = self.tcfg.get("save_hf_interval", None)
        if save_hf_interval is None:
            save_hf_interval = max(50, int(total_opt_steps * self.tcfg.get("save_hf_ratio", 0.1)))

        # Resume checkpoint interval
        if self._save_interval_cfg is None:
            self.save_interval = max(50, int(total_opt_steps * self.tcfg.get("save_ratio", 0.2)))

        self.acc.print(f"[train] total_opt_steps={total_opt_steps}, save_interval={self.save_interval}, "
                       f"save_hf_interval={save_hf_interval}, eval_interval={eval_interval}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            it = self.train_loader
            it = tqdm(it, desc=f"Epoch {epoch}", disable=not self.acc.is_main_process)

            # Skip batches if resuming mid-epoch (only on first iteration)
            if not resume_skip_done and self.current_step > 0:
                # current_step counts optimizer steps; each consumes grad_accum batches
                total_batches_consumed = self.current_step * gradient_accumulation_steps
                batches_to_skip = total_batches_consumed % steps_per_epoch
                if batches_to_skip > 0:
                    self.acc.print(f"[resume] Skipping {batches_to_skip} batches (step {self.current_step} x {gradient_accumulation_steps} grad_accum) in epoch {epoch}")
                    for skip_idx, _ in enumerate(it):
                        if skip_idx >= batches_to_skip - 1:
                            break
                resume_skip_done = True
            
            # Normal training loop with gradient accumulation
            for batch_idx, (x, y) in enumerate(it):
                # Use accumulate context manager for gradient accumulation
                with self.acc.accumulate(self.model):
                    output = self.model(input_ids=x, labels=x)
                    loss = output.loss
                    self.acc.backward(loss)
                    
                    # Only step optimizer when we've accumulated enough gradients
                    if self.acc.sync_gradients:
                        # Optional: gradient clipping for stability
                        max_grad_norm = self.tcfg.get("max_grad_norm", None)
                        if max_grad_norm is not None:
                            self.acc.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                        
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.current_step += 1
                        self._last_loss = loss.item()

                        # Log only on actual optimization steps
                        if self.current_step % self.log_interval == 0:
                            self._log(epoch, self.current_step, loss)
                        
                        # Eval: only loss + entropy on held-out data
                        if self.current_step == 10 or self.current_step % eval_interval == 0:
                            self.acc.wait_for_everyone()
                            if self.eval_loader is not None and self.acc.is_main_process:
                                val = self._evaluate(current_step=self.current_step, max_steps=self.cfg.training.get("eval_max_steps", 50))
                                if val is not None:
                                    self.acc.print(f"[eval] step {self.current_step} " + " ".join([f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, float)]))
                                    self.acc.log({f"eval/{k}": v for k, v in val.items() if 'num' not in k}, step=self.current_step)
                                    self._write_local_log({
                                        "type": "eval", "step": self.current_step,
                                        **{k: v for k, v in val.items() if isinstance(v, float)},
                                    })
                            self.acc.wait_for_everyone()
                            self.model.train()
                            
                        if self.current_step % self.save_interval == 0:
                            self._save_ckpt(self.current_step)

                        if self.current_step % save_hf_interval == 0:
                            self.acc.wait_for_everyone()
                            self._save_hf_ckpt(self.current_step)
                            self.acc.wait_for_everyone()

                        # Stop if we've reached the target steps (isoflop budget)
                        if max_steps is not None and self.current_step >= max_steps:
                            self.acc.print(f"[train] Reached max_steps={max_steps}, stopping.")
                            break

                if max_steps is not None and self.current_step >= max_steps:
                    break

            # Recreate dataloader for next epoch with deterministic seed
            self.acc.wait_for_everyone()
            self.train_loader = create_dataloader(
                txt_files=self.train_files,
                tokenizer=self.tok,
                batch_size=self.tcfg.batch_size,
                seq_len=self.mcfg.block_size,
                num_workers=self.dataloader_kwargs["num_workers"],
                shuffle=False,
                cache_size=self._cache_size,
                dataset_shuffle=True,
                num_shards=self.dcfg.get("num_shards", None),
                prefetch_factor=self.dataloader_kwargs["prefetch_factor"],
                persistent_workers=self.dataloader_kwargs["persistent_workers"],
                seed=self._data_seed + epoch + 1,
            )
            self.train_loader = self.acc.prepare(self.train_loader)

        # Final clean HF model + last resume checkpoint
        self.acc.wait_for_everyone()

        # Always log final step — the break above skips interval-based logging
        if self.acc.is_main_process and hasattr(self, '_last_loss'):
            self._write_local_log({
                "type": "train", "step": self.current_step, "epoch": self.current_epoch,
                "loss": self._last_loss, "lr": self.scheduler.get_last_lr()[0],
            })
        if self.eval_loader is not None and self.acc.is_main_process:
            val = self._evaluate(current_step=self.current_step, max_steps=self.cfg.training.get("eval_max_steps", 50))
            if val is not None:
                self.acc.print(f"[eval] final step {self.current_step} " + " ".join([f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, float)]))
                self.acc.log({f"eval/{k}": v for k, v in val.items() if 'num' not in k}, step=self.current_step)
                self._write_local_log({
                    "type": "eval", "step": self.current_step,
                    **{k: v for k, v in val.items() if isinstance(v, float)},
                })

        self._save_ckpt(self.current_step)
        self._final_ckpt(self.current_step)
        # Wait for any in-flight HF upload to complete before exiting
        if self._hf_upload_thread is not None and self._hf_upload_thread.is_alive():
            self.acc.print("[hf_upload] Waiting for upload to finish...")
            self._hf_upload_thread.join()
        self.acc.end_training()

    @torch.no_grad()
    def _evaluate(self, current_step: int, max_steps: int | None = None):
        metrics = {}
        if self.eval_loader is not None:
            base_model = self.acc.unwrap_model(self.model)
            base_model.eval()
            total_loss, total_entropy, n = 0.0, 0.0, 0
            for i, (x, y) in enumerate(self.eval_loader):
                x = x.to(self.acc.device)
                output = base_model(input_ids=x, labels=x)
                total_loss += output.loss.item()
                # per-token entropy from logits
                probs = torch.softmax(output.logits.view(-1, output.logits.size(-1)), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                total_entropy += entropy
                n += 1
                if max_steps and i+1 >= max_steps: break
            metrics['loss'] = total_loss / max(n, 1)
            metrics['entropy_per_token'] = total_entropy / max(n, 1)
        
        return metrics
    
    @torch.no_grad()
    def _compute_test_entropy(self, evaluator, file_path, model, max_samples=100):
        """Compute move-level entropy and top-k probability mass on test positions.

        For each position, gets all legal moves, scores each via a batched
        teacher-forced forward pass, normalises to a distribution over legal
        moves, and computes entropy (nats) and top-k cumulative probability.

        Returns dict with keys: entropy, top1_mass, top5_mass
        """
        df = evaluator.load_data(file_path)
        state_col = evaluator.get_state_column()
        move_col = evaluator.get_move_column()
        df = df.dropna(subset=[state_col, move_col]).head(max_samples)
        if len(df) == 0:
            return None

        device = self.acc.device
        top_ks = [1, 5]
        total_entropy, n_positions = 0.0, 0
        total_topk = {k: 0.0 for k in top_ks}

        for _, row in df.iterrows():
            state_text = str(row[state_col]).strip()
            try:
                board = self._board_from_pgn(state_text)
            except Exception:
                continue
            legal_moves = list(board.legal_moves)
            if len(legal_moves) == 0:
                continue

            # Convert each legal move to LAN and tokenize
            state_prefix_ids = self.tok.encode(state_text + " ")
            prefix_len = len(state_prefix_ids)
            sequences = []
            for mv in legal_moves:
                lan = self._move_to_lan(board, mv)
                if not lan:
                    continue
                full_ids = self.tok.encode(state_text + " " + lan)
                sequences.append(full_ids)

            if len(sequences) == 0:
                continue

            # Pad and batch all candidates
            max_len = min(max(len(s) for s in sequences), self.mcfg.block_size)
            padded = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
            for j, seq in enumerate(sequences):
                sl = min(len(seq), max_len)
                padded[j, :sl] = torch.tensor(seq[:sl], dtype=torch.long)

            # Single batched forward pass for all legal moves of this position
            output = model(input_ids=padded)
            logits = output.logits  # (num_legal_moves, T, V)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Score each move: sum log P(token_t | prefix + tokens<t) over move tokens
            move_log_probs = []
            for j, seq in enumerate(sequences):
                sl = min(len(seq), max_len)
                ms = prefix_len - 1  # position whose logits predict the first move token
                if ms >= sl - 1:
                    move_log_probs.append(torch.tensor(0.0, device=device))
                    continue
                # gather log probs for each move token
                target_ids = padded[j, ms + 1 : sl]  # (move_len,)
                pred_log_probs = log_probs[j, ms : sl - 1]  # (move_len, V)
                token_lps = pred_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                move_log_probs.append(token_lps.sum())

            # Normalise over legal moves → proper distribution
            move_log_probs_t = torch.stack(move_log_probs)  # (num_legal_moves,)
            move_probs = torch.softmax(move_log_probs_t, dim=0)  # normalised

            # Entropy
            entropy = -(move_probs * torch.log(move_probs + 1e-9)).sum().item()
            total_entropy += entropy

            # Top-k probability mass
            sorted_probs, _ = move_probs.sort(descending=True)
            for k in top_ks:
                total_topk[k] += sorted_probs[:k].sum().item()

            n_positions += 1

        if n_positions == 0:
            return None
        result = {"entropy": total_entropy / n_positions}
        for k in top_ks:
            result[f"top{k}_mass"] = total_topk[k] / n_positions
        return result

    @torch.no_grad()
    def _evaluate_kl_divergence(self, current_step: int, max_steps: int | None = None, stockfish_path: str = "${REPO_ROOT}/stockfish", stockfish_time: float = 0.1):
        """
        Evaluate normalized KL divergence between model move distribution and engine evaluation.
        
        For each position in the eval dataset:
        1. Parse moves from the original sequence
        2. Get predicted moves from model
        3. Evaluate all legal moves on current state with engine
        4. Calculate normalized KL divergence between model and engine distributions
        
        Args:
            current_step: Current training step
            max_steps: Maximum number of batches to evaluate
            stockfish_path: Path to Stockfish executable
            stockfish_time: Time limit for Stockfish evaluation per move
        
        Returns:
            Dictionary with KL divergence metrics
        """
        import chess
        import chess.engine
        import chess.pgn
        import io
        import math
        from scipy.stats import entropy
        
        metrics = {}
        if self.eval_loader is None:
            return metrics
        
        self.model.eval()
        
        # Initialize Stockfish engine
        try:
            engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        except Exception as e:
            self.acc.print(f"[warn] Failed to initialize Stockfish: {e}")
            return metrics
        
        total_kl = 0.0
        total_positions = 0
        valid_positions = 0
        
        try:
            for i, (x, y) in enumerate(self.eval_loader):
                if max_steps and i + 1 > max_steps:
                    break
                
                # Decode sequences to get original text
                batch_size = x.shape[0]
                for batch_idx in range(batch_size):
                    sequence_ids = x[batch_idx].cpu().tolist()
                    # Remove padding tokens (0 or pad_id)
                    pad_id = self.tok.pad_id() if hasattr(self.tok, 'pad_id') and self.tok.pad_id() is not None else 0
                    sequence_ids = [tid for tid in sequence_ids if tid != pad_id]
                    sequence_text = self.tok.decode(sequence_ids)
                    
                    # Parse sequence to extract positions and moves
                    try:
                        # Try parsing as PGN
                        game = chess.pgn.read_game(io.StringIO(sequence_text))
                        if game is None:
                            continue
                        
                        board = game.board()
                        moves_played = list(game.mainline_moves())
                        
                        if len(moves_played) == 0:
                            continue
                        
                        # Process each position in the game
                        for move_idx, move in enumerate(moves_played):
                            # Get current position before this move
                            current_board = board.copy()
                            
                            # Get all legal moves at this position
                            legal_moves = list(current_board.legal_moves)
                            if len(legal_moves) == 0:
                                board.push(move)
                                continue
                            
                            # Get state string (PGN up to this point)
                            state_text = self._board_to_pgn_string(current_board)
                            
                            # Get model logits for all legal moves
                            model_probs = self._get_model_move_probs(state_text, legal_moves)
                            if model_probs is None or len(model_probs) != len(legal_moves):
                                board.push(move)
                                continue
                            
                            # Get engine evaluations for all legal moves
                            engine_scores = []
                            for legal_move in legal_moves:
                                test_board = current_board.copy()
                                test_board.push(legal_move)
                                try:
                                    info = engine.analyse(test_board, chess.engine.Limit(time=stockfish_time))
                                    # Get score from perspective of current player
                                    score = info["score"].pov(current_board.turn).score(mate_score=100000)
                                    engine_scores.append(float(score))
                                except Exception:
                                    engine_scores.append(0.0)
                            
                            # Convert engine scores to probability distribution (softmax)
                            if len(engine_scores) == len(legal_moves) and len(engine_scores) > 0:
                                # Normalize scores using softmax
                                max_score = max(engine_scores)
                                exp_scores = [math.exp(s - max_score) for s in engine_scores]
                                sum_exp = sum(exp_scores)
                                if sum_exp > 0:
                                    engine_probs = [e / sum_exp for e in exp_scores]
                                    
                                    # Calculate KL divergence: KL(model || engine)
                                    # Add small epsilon to avoid log(0)
                                    epsilon = 1e-10
                                    model_probs_safe = [max(p, epsilon) for p in model_probs]
                                    engine_probs_safe = [max(p, epsilon) for p in engine_probs]
                                    
                                    # Normalize to ensure they sum to 1
                                    model_sum = sum(model_probs_safe)
                                    engine_sum = sum(engine_probs_safe)
                                    if model_sum > 0 and engine_sum > 0:
                                        model_probs_safe = [p / model_sum for p in model_probs_safe]
                                        engine_probs_safe = [p / engine_sum for p in engine_probs_safe]
                                        
                                        # Calculate KL divergence
                                        kl_div = entropy(model_probs_safe, engine_probs_safe)
                                        
                                        # Normalize KL by log(number of moves) to get normalized KL
                                        num_moves = len(legal_moves)
                                        normalized_kl = kl_div / math.log(num_moves) if num_moves > 1 else kl_div
                                        
                                        total_kl += normalized_kl
                                        valid_positions += 1
                            
                            # Advance board
                            board.push(move)
                        
                        total_positions += len(moves_played)
                    
                    except Exception as e:
                        # Skip sequences that can't be parsed
                        continue
        
        finally:
            engine.quit()
        
        self.model.train()
        
        if valid_positions > 0:
            avg_kl = total_kl / valid_positions
            metrics['kl_divergence'] = avg_kl
            metrics['num_positions'] = valid_positions
            metrics['total_positions'] = total_positions
        else:
            metrics['kl_divergence'] = 0.0
            metrics['num_positions'] = 0
            metrics['total_positions'] = total_positions
        
        return metrics
    
    def _board_to_pgn_string(self, board) -> str:
        """Convert a chess board to a PGN string representation."""
        import chess
        import chess.pgn
        game = chess.pgn.Game.from_board(board)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        pgn_str = game.accept(exporter)
        # Remove headers and result markers
        import re
        pgn_str = re.sub(r"\[[^\]]*\]\s*", "", pgn_str)
        pgn_str = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", pgn_str).strip()
        return pgn_str
    
    def _get_model_move_probs(self, state_text: str, legal_moves: list) -> list | None:
        """
        Get model probability distribution over legal moves.
        Computes probability of each move by generating it token by token.
        
        Args:
            state_text: Current state as PGN string
            legal_moves: List of legal chess.Move objects
        
        Returns:
            List of probabilities for each legal move, or None if failed
        """
        import chess
        import chess.pgn
        import io
        from evaluation.utils import lan_to_uci
        
        try:
            # Reconstruct board to get move in proper format
            try:
                game = chess.pgn.read_game(io.StringIO(state_text))
                if game:
                    board = game.end().board()
                else:
                    board = chess.Board()
            except:
                board = chess.Board()
            
            # Encode the state
            state_ids = self.tok.encode(state_text)
            if self.tok.eos_id() is not None and state_ids and state_ids[-1] == self.tok.eos_id():
                state_ids = state_ids[:-1]
            
            # For each legal move, compute its probability
            move_probs = []
            for move in legal_moves:
                try:
                    # Get move in LAN format (as the model would generate it)
                    # The model uses LAN format like "Pd2d4", "Pd4xe5", etc.
                    move_lan = self._move_to_lan(board, move)
                    if not move_lan:
                        move_probs.append(1e-10)
                        continue
                    
                    # Compute probability of this move sequence
                    # Encode state + move to see how it's tokenized
                    state_plus_move = state_text + " " + move_lan
                    move_ids = self.tok.encode(move_lan)
                    
                    if not move_ids:
                        move_probs.append(1e-10)
                        continue
                    
                    # Compute probability token by token
                    current_ids = state_ids.copy()
                    move_prob = 1.0
                    
                    for token_idx, token_id in enumerate(move_ids):
                        input_tensor = torch.tensor([current_ids], dtype=torch.long, device=self.acc.device)
                        
                        with torch.no_grad():
                            outputs = self.model(input_ids=input_tensor)
                            logits = outputs.logits[0, -1, :]  # Last position
                            probs = torch.softmax(logits, dim=-1)
                            
                            if token_id < len(probs):
                                token_prob = probs[token_id].item()
                                move_prob *= token_prob
                            else:
                                move_prob = 1e-10
                                break
                        
                        # Add this token to context for next iteration
                        current_ids.append(token_id)
                    
                    move_probs.append(max(move_prob, 1e-10))
                
                except Exception:
                    move_probs.append(1e-10)
            
            # Normalize probabilities
            if move_probs and sum(move_probs) > 0:
                total = sum(move_probs)
                move_probs = [p / total for p in move_probs]
                return move_probs
            
            return None
        
        except Exception as e:
            return None
    
    def _move_to_lan(self, board, move) -> str:
        """
        Convert a chess.Move to LAN format (e.g., "Pd2d4", "Pd4xe5").
        This matches the format used by the tokenizer.
        """
        import chess
        try:
            # Get piece symbol
            piece = board.piece_at(move.from_square)
            if piece is None:
                return ""
            
            piece_symbol = piece.symbol().upper()
            
            # Handle castling
            if board.is_castling(move):
                if move.to_square == chess.G1 or move.to_square == chess.G8:
                    return "O-O"
                else:
                    return "O-O-O"
            
            # Get squares
            from_sq = chess.square_name(move.from_square)
            to_sq = chess.square_name(move.to_square)
            
            # Check if capture
            is_capture = board.is_capture(move)
            capture_str = "x" if is_capture else ""
            
            # Check promotion
            promo_str = ""
            if move.promotion:
                promo_pieces = {chess.QUEEN: "Q", chess.ROOK: "R", chess.BISHOP: "B", chess.KNIGHT: "N"}
                promo_str = "=" + promo_pieces.get(move.promotion, "Q")
            
            # Build LAN string: Piece + from_square + (x) + to_square + (=promo)
            lan = f"{piece_symbol}{from_sq}{capture_str}{to_sq}{promo_str}"
            return lan
        
        except Exception:
            return ""

    @torch.no_grad()
    def evaluate_pass_at_k(self, k: int = 5, max_samples: int = 200):
        """
        Run pass@k evaluation on all configured evaluation datasets.
        
        Args:
            k: Number of samples to generate per state for pass@k evaluation
            max_samples: Maximum number of samples to evaluate per dataset
        
        Returns:
            Dictionary of metrics for all evaluation datasets
        """
        metrics = {}
        
        base_model = self.acc.unwrap_model(self.model)
        base_model.eval()
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        
        # Create evaluators fresh with current model
        evaluator_configs = {
            "model": base_model, 
            "tokenizer": self.tok,
            "device": self.acc.device,
            "move_format": "uci",
            "batch_size": 32,
            "max_new_tokens": 20,
            "temperature": 1.0,
            "top_k": None
        }
        
        if not self.acc.is_main_process:
            return metrics
        
        self.acc.print(f"\n{'='*60}")
        self.acc.print(f"Running PASS@{k} Evaluation")
        self.acc.print(f"{'='*60}\n")
        
        test_data_dir = self.dcfg.get("test_data_dir", "${REPO_ROOT}/data")
        _pass_at_k_table = [
            (HumanGamesEvaluator, "chess/test_large/human_games_test_opening.parquet", "human_opening"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_capture.parquet", "human_capture"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_random.parquet", "human_random"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_promotion.parquet", "human_promotion"),
            (HumanGamesEvaluator, "chess/test_large/human_games_test_check.parquet", "human_check"),
            (PuzzlesEvaluator, "chess/test/puzzles_grandmaster.csv", "puzzles_grandmaster"),
            (PuzzlesEvaluator, "chess/test/puzzles_test.csv", "puzzles_test"),
            (RandomGamesEvaluator, "chess/test/random_games_100.parquet", "random_games"),
        ]

        eval_configs = []
        for eval_cls, rel_path, prefix in _pass_at_k_table:
            file_path = os.path.join(test_data_dir, rel_path)
            if not os.path.exists(file_path):
                continue
            eval_configs.append({
                "evaluator": eval_cls(**evaluator_configs),
                "file": file_path,
                "prefix": prefix,
            })

        if not eval_configs:
            self.acc.print(f"[pass@k] No test files found under {test_data_dir}, skipping")
            return metrics
        
        for config in eval_configs:
            self.acc.print(f"\nEvaluating {config['prefix']}...")
            pred_dir = f"{self.save_dir}/rollouts/pass_at_k/{config['prefix']}"
            os.makedirs(pred_dir, exist_ok=True)

            output_path = f"{pred_dir}/predictions_pass_at_{k}.parquet"
            output_metrics_path = f"{pred_dir}/metrics_pass_at_k.jsonl"
            
            try:
                eval_metrics = config["evaluator"].evaluate_pass_at_k(
                    config["file"],
                    k=k,
                    max_samples=max_samples,
                    verbose=True,
                    output_path=output_path
                )
                
                import json
                record = {"k": k, "prefix": config["prefix"], **eval_metrics}
                with open(output_metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                metrics.update({
                    f"{config['prefix']}_{k}": v
                    for k, v in eval_metrics.items()
                    if "num" not in k
                })
                
                self.acc.print(f"✓ {config['prefix']}: Pass@{k} = {eval_metrics.get('pass_at_k', 0):.2%}")
                self.acc.print(f"  Saved to {output_path}")
                
            except Exception as e:
                self.acc.print(f"✗ Error evaluating {config['prefix']}: {e}")
                import traceback
                traceback.print_exc()
        
        self.acc.print(f"\n{'='*60}")
        self.acc.print(f"PASS@{k} Evaluation Complete")
        self.acc.print(f"{'='*60}\n")
        
        return metrics

    def _write_local_log(self, record: dict):
        """Append a JSON line to the local metrics log file (main process only)."""
        if not self.acc.is_main_process:
            return
        with open(self._local_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _log(self, epoch, step, loss):
        if self.acc.is_main_process:
            lr = self.scheduler.get_last_lr()[0]
            now = time.time()
            elapsed = now - self._last_log_time
            steps_delta = step - self._last_log_step
            grad_accum = self.tcfg.get("gradient_accumulation_steps", 1)
            # step is now in optimizer-step units; each opt step processes
            # grad_accum batches * num_processes ranks * tokens_per_batch tokens
            tokens_processed = steps_delta * self._tokens_per_batch * grad_accum * self.acc.num_processes
            tokens_per_sec = tokens_processed / max(elapsed, 1e-9)
            achieved_tflops = (tokens_per_sec * self._flops_per_token) / 1e12
            mfu = achieved_tflops / self._gpu_peak_tflops

            self._last_log_time = now
            self._last_log_step = step

            self.acc.print(
                f"epoch {epoch} step {step} loss={loss.item():.4f} lr={lr:.2e} "
                f"tok/s={tokens_per_sec:.0f} mfu={mfu:.3f}"
            )
            self.acc.log({
                "train/loss": loss.item(), "lr": lr, "epoch": epoch,
                "train/tokens_per_sec": tokens_per_sec,
                "train/tflops": achieved_tflops,
                "train/mfu": mfu,
            }, step=step)
            self._write_local_log({
                "type": "train", "step": step, "epoch": epoch,
                "loss": loss.item(), "lr": lr,
            })

    def _save_ckpt(self, step):
        """Save lean resume checkpoint to latest/ (overwritten each time)."""
        out = self.save_dir / "latest"
        if self.acc.is_main_process:
            out.mkdir(parents=True, exist_ok=True)
        self.acc.wait_for_everyone()
        self.acc.save_state(str(out))
        self._save_training_state(out)
        if self.acc.is_main_process:
            self.acc.print(f"[ckpt] resume checkpoint saved → {out}  (step {step})")
            self._upload_run_dir_to_hf(f"resume_step_{step}")

    def _save_hf_ckpt(self, step):
        """Save clean HF model snapshot (weights + tokenizer + generation_config)."""
        if not self.acc.is_main_process:
            return
        out = self.save_dir / f"step_{step}"
        out.mkdir(parents=True, exist_ok=True)
        self._save_hf_model(out)
        self._save_generation_config(out)
        self.acc.print(f"[ckpt] HF model snapshot saved → {out}")
        self._upload_run_dir_to_hf(f"step_{step}")

    def _final_ckpt(self, step):
        """Save final clean HF model (for upload / inference)."""
        if not self.acc.is_main_process:
            return
        out = self.save_dir / "final"
        out.mkdir(parents=True, exist_ok=True)
        self._save_hf_model(out)
        self._save_generation_config(out)
        self.acc.print(f"[ckpt] final HF model saved → {out}")
        self._upload_run_dir_to_hf("final")

    def _upload_run_dir_to_hf(self, step_label):
        """Upload the full run directory to HuggingFace Hub in a background thread.

        On the first call of each training run the remote repo is deleted and
        recreated from scratch so stale files from a previous experiment never
        linger.  Subsequent calls (intermediate checkpoints) just sync on top.
        The thread is non-daemon so the OS does not kill it when the main
        process exits; train() also joins it explicitly before end_training().
        """
        if not self.acc.is_main_process:
            return
        hf_repo = self.tcfg.get("hf_upload_repo", None)
        if not hf_repo:
            return

        import threading
        from huggingface_hub import HfApi

        # Capture whether we need to clear so the closure is consistent even
        # if _hf_repo_cleared changes before the thread runs.
        should_clear = not self._hf_repo_cleared
        if should_clear:
            self._hf_repo_cleared = True

        # Wait for any previous upload thread to finish before starting a new one
        if self._hf_upload_thread is not None and self._hf_upload_thread.is_alive():
            self.acc.print(f"[hf_upload] Waiting for previous upload before starting {step_label}...")
            self._hf_upload_thread.join()
        # Previous upload is complete — safe to delete old checkpoints now
        self._cleanup_old_checkpoints()

        def _do_upload():
            try:
                api = HfApi()
                if should_clear:
                    # Delete the repo if it exists so old model files don't linger
                    try:
                        api.delete_repo(hf_repo, repo_type="model")
                        self.acc.print(f"[hf_upload] Cleared existing repo {hf_repo}")
                    except Exception:
                        pass  # repo didn't exist yet — that's fine
                api.create_repo(hf_repo, exist_ok=True, repo_type="model")
                api.upload_folder(
                    folder_path=str(self.save_dir),
                    repo_id=hf_repo,
                    path_in_repo="",
                    commit_message=f"Sync full run dir at {step_label}",
                )
                self.acc.print(f"[hf_upload] synced full run dir ({step_label}) -> {hf_repo}")
            except Exception as e:
                self.acc.print(f"[hf_upload] Failed full run dir sync at {step_label}: {e}")

        t = threading.Thread(target=_do_upload, daemon=False)
        t.start()
        self._hf_upload_thread = t
    
    def _save_hf_model(self, path):
        """Save model + tokenizer in HuggingFace format (from_pretrained compatible)."""
        from sft.training.hf_tokenizer_utils import save_hf_tokenizer
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Unwrap model from accelerator
        base_model = self.acc.unwrap_model(self.model)

        base_model.config.vocab_size = self.vocab_size
        base_model.config.bos_token_id = self.tok.bos_id()
        base_model.config.eos_token_id = self.tok.eos_id()
        base_model.config.pad_token_id = self.tok.pad_id()

        # Save model weights + config manually to avoid Qwen3 layer_type validation
        import json as _json
        from safetensors.torch import save_file as _save_safetensors

        # Save weights (clone tied tensors to avoid shared memory error)
        state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}
        _save_safetensors(state_dict, str(pathlib.Path(path) / "model.safetensors"))

        # Save config with synchronized layer metadata to avoid arch validation issues.
        config_dict = base_model.config.to_dict()

        # Prefer explicit training config; fall back to serialized config fields.
        n_layers = self.mcfg.get("n_layer", None)
        if n_layers is None:
            n_layers = config_dict.get("num_hidden_layers", config_dict.get("n_layer", 28))
        n_layers = int(n_layers)

        if "num_hidden_layers" in config_dict:
            config_dict["num_hidden_layers"] = n_layers
        if "n_layer" in config_dict:
            config_dict["n_layer"] = n_layers

        def _normalize_layer_list(values, target_len):
            if target_len <= 0:
                return []
            if not isinstance(values, list):
                return ["full_attention"] * target_len
            if len(values) >= target_len:
                return values[:target_len]
            fill_value = values[-1] if values and isinstance(values[-1], str) else "full_attention"
            return values + [fill_value] * (target_len - len(values))

        for key in ("layer_type", "layer_types"):
            if key in config_dict:
                config_dict[key] = _normalize_layer_list(config_dict[key], n_layers)

        # Keep this field aligned with hidden layers for downstream checks/loaders.
        if "max_window_layers" in config_dict:
            config_dict["max_window_layers"] = n_layers
        with open(pathlib.Path(path) / "config.json", "w") as f:
            _json.dump(config_dict, f, indent=2)

        # Save self-contained tokenizer (vocab.json, tokenizer.py, etc.)
        save_hf_tokenizer(
            tokenizer=self.tok,
            tokcfg=self.tokcfg,
            save_directory=path,
            model_max_length=self.mcfg.get("block_size", 2048),
        )

        self.acc.print(f"[train] HuggingFace model + tokenizer saved → {path}")

    def _save_generation_config(self, path):
        """Save generation_config.json for inference."""
        import json
        gen_config = {
            "bos_token_id": self.tok.bos_id(),
            "eos_token_id": self.tok.eos_id(),
            "pad_token_id": self.tok.pad_id(),
            "max_new_tokens": self.mcfg.get("block_size", 1024),
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
        }
        with open(pathlib.Path(path) / "generation_config.json", "w") as f:
            json.dump(gen_config, f, indent=2)

    def _expand_txt_files(self, dcfg):
        """Return a sorted list[Path] of training shard files from config."""
        if "txt_files" in dcfg and dcfg.get("txt_files"):
            items = dcfg.get("txt_files")
        elif "txt_path" in dcfg and dcfg.get("txt_path"):
            items = dcfg.get("txt_path")
        else:
            raise ValueError("Config must provide data.txt_files or data.txt_path")

        uniq = self._expand_paths(items)
        if not uniq:
            raise FileNotFoundError("No shards found from data paths.")
        return uniq

    def _expand_paths(self, items):
        """Expand dirs / globs / single files into a de-duplicated list[Path]."""
        import glob
        import os
        from pathlib import Path
        def expand_one(p):
            p = Path(p)
            if p.is_dir():
                # nested shard_* subdirectory structure takes priority
                shard_dirs = sorted(p.glob("shard_*"))
                if shard_dirs:
                    files = []
                    for sd in shard_dirs:
                        files.extend(sorted(sd.glob("*.npy")))
                    return files
                return sorted(p.glob("*.txt")) + sorted(p.glob("*.npy"))
            # allow simple glob strings too, e.g. '/data/chess/raw.*.txt'
            # (glob.glob handles absolute patterns; Path().glob does not)
            if any(ch in str(p) for ch in "*?[]"):
                return [Path(m) for m in sorted(glob.glob(str(p)))]
            # single file
            return [p] if p.exists() else []

        if isinstance(items, (str, Path)):
            items = [items]
        out = []
        for it in items:
            out.extend(expand_one(it))

        # de-dup while preserving insertion order (shard_dir → filename order)
        # abspath, not resolve(): resolve() realpath()s every path component, and
        # on an NFS mount with lookupcache=none that is a server round-trip per
        # component (~30ms/file measured). Across ~47k shards x 8 ranks that is
        # ~20min of startup before the first step. abspath is pure string work.
        seen = set()
        uniq = []
        for p in (Path(os.path.abspath(p)) for p in out):
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq
