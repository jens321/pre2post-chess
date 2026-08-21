# Speed tests: max per-device batch size on 8x B200 (183 GB), seq_len 1024, bf16.
#
# The pretrain_tokens override is required, not cosmetic: the configs carry a
# 100M-token stub, which at these batch sizes is 15-200 optimizer steps and
# would exit before the second log line. save_interval is pushed out so no
# checkpoints are written mid-measurement.
#
# Measured peaks (largest batch that fits) and what to expect:
#   model   params    bs    tok/s     mem/GPU    util   next size up
#    20m     20.50M   768   8.58M     166.0 GB   100%   1024 -> OOM
#   100m    101.53M   256   2.195M    175.6 GB   100%    384 -> OOM
#   680m    678.49M    64   516.6K    170.5 GB   100%     96 -> OOM
#
# params is what the trainer actually builds, not what the config header claims.
# The 100m/680m sweep configs have headers that disagree with their own bodies
# (kv_heads, n_layer); the bodies are what run and what is mirrored here.
#
# Throughput saturates well below these peaks (20m: 128 gives 93%; 100m: 128
# gives 98%; 680m: 32 gives 95%). The peaks sit at 90-96% of HBM and still fit:
# the step-10 eval ran at every batch size above without OOM, since eval is
# no-grad and stores no backward activations.
#
# EVAL POINTS: eval_interval=100 is set explicitly below so wandb gets a real
# eval curve (steps 10, 100, 200, ...). Without it, eval_interval defaults to
# max(10, 0.1 * total_opt_steps) and the pretrain_tokens override inflates
# total_opt_steps so far that the second eval would land at step 1589 / 4768 /
# 19073 (~19 min / ~76 min / ~5.4 h) -- one point and then nothing.
#
# Cost: each eval is eval_max_steps=50 forward-only batches, very roughly 15% of
# wall clock at this interval. Whichever log line contains an eval reads low,
# because tok/s is a windowed rate -- take throughput from the other lines, or
# drop eval_interval for a clean measurement. Lower eval_max_steps to cheapen it.
#
# Logged `mfu` is aggregate: total achieved TFLOPS across all ranks divided by
# ONE GPU's peak. Divide by NUM_GPUS for true per-GPU MFU (20m: 5.9%, 100m:
# 7.4%, 680m: 11.7%).

# 20m model speed test
WANDB_MODE=online CONFIG=config/configs/20m_test.yaml DATA_DIR=/data/home/jtuyls/data/pre2post-chess/pretrain_v1_20b OUTPUT_DIR=/tmp/bs768_repro NUM_GPUS=8 OVERRIDES="data.pretrain_tokens=100000000000 training.save_interval=100000000 training.log_interval=10 training.eval_interval=200 training.compile_model=true training.compile_mode=default" bash run_pretrain.sh

# 100m model speed test
WANDB_MODE=online CONFIG=config/configs/100m_test.yaml DATA_DIR=/data/home/jtuyls/data/pre2post-chess/pretrain_v1_20b OUTPUT_DIR=/tmp/bs256_repro NUM_GPUS=8 OVERRIDES="data.pretrain_tokens=100000000000 training.save_interval=100000000 training.log_interval=10 training.eval_interval=200 training.compile_model=true training.compile_mode=default" bash run_pretrain.sh

# 680m model speed test
WANDB_MODE=online CONFIG=config/configs/680m_test.yaml DATA_DIR=/data/home/jtuyls/data/pre2post-chess/pretrain_v1_20b OUTPUT_DIR=/tmp/bs64_repro NUM_GPUS=8 OVERRIDES="data.pretrain_tokens=100000000000 training.save_interval=100000000 training.log_interval=10 training.eval_interval=200 training.compile_model=true training.compile_mode=default" bash run_pretrain.sh
