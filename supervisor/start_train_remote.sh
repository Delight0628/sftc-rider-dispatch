#!/bin/bash
# Robust MAPPO training launcher (unbuffered, PID file, metrics).
set -u
ROOT=/gemini/code/sftc-rider-dispatch
LOGDIR=/quota/train_logs
MODELDIR=/quota/models
LOGFILE="$LOGDIR/train_full.log"
PIDFILE="$LOGDIR/train.pid"
mkdir -p "$LOGDIR" "$MODELDIR" "$LOGDIR/episodes"

# 训练日志目录结构（参考 MARL_FOR_W_Factory）:
#   $LOGDIR/train_full.log          # 完整训练 stdout
#   $LOGDIR/metrics.jsonl           # 每回合结构化指标
#   $LOGDIR/run_*/episodes/ep_XXXXXX/  # 每回合完整日志
#       episode.log  metrics.json

# stop previous
if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "stopping old train pid=$old"
    kill "$old" 2>/dev/null || true
    sleep 2
    kill -9 "$old" 2>/dev/null || true
  fi
fi
pkill -f ppo_marl_train.py 2>/dev/null || true
pkill -f 'spawn_main' 2>/dev/null || true
sleep 1

# bump parallel workers for this CPU box (96 cores)
python3 - <<'PY'
from pathlib import Path
p = Path("/gemini/code/sftc-rider-dispatch/environments/w_factory_config.py")
text = p.read_text(encoding="utf-8")
old = '"num_parallel_workers": 4,'
new = '"num_parallel_workers": 8,'
if old in text:
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("workers -> 8")
else:
    print("workers config unchanged")
PY

cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}

echo "starting training at $(date)" | tee -a "$LOGFILE"
nohup /root/miniconda3/bin/python -u mappo/ppo_marl_train.py \
  --scenario delivery \
  --candidate-source upstream \
  --models-dir "$MODELDIR" \
  --logs-dir /quota/train_logs \
  >> "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "TRAIN_PID=$PID"
sleep 6
ps -p "$PID" -o pid,etime,cmd --no-headers || echo "MAIN NOT RUNNING"
pgrep -af 'ppo_marl_train|spawn_main' | head
echo "--- log tail ---"
tail -30 "$LOGFILE"
