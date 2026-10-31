#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/project/xiongxiaobin/ky26212/rowing
export OMP_NUM_THREADS=4
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

CHECKPOINT=/inspire/hdd/project/xiongxiaobin/ky26212/rowing/logs/rowing_multigpu/run01/rank_0/model_17700.pt
LOG_DIR=/inspire/hdd/project/xiongxiaobin/ky26212/rowing/logs/rowing_finetune_h200/run01
test -f "$CHECKPOINT"
mkdir -p "$LOG_DIR"
nvidia-smi
python3 -c 'import torch; assert torch.cuda.device_count() == 1, "Expected one visible H200"; print(torch.__version__, torch.cuda.get_device_name(0))'
python3 -m rowing_mjlab.finetune \
  --device cuda:0 \
  --num-envs 1024 \
  --iterations 3000 \
  --checkpoint "$CHECKPOINT" \
  --log-dir "$LOG_DIR" \
  --compile
