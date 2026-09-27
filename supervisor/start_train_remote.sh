#!/bin/bash
# Launch MAPPO training with artifacts under the CODE path (persist across job restarts).
# Layout follows MARL_FOR_W_Factory: models + logs live next to the code.
set -u

CODE_ROOT=/gemini/code/sftc-rider-dispatch
STAMP=$(date +%m%d_%H%M)
RUN_NAME="${STAMP}_fulltrain"
RUN_DIR="${CODE_ROOT}/runs/${RUN_NAME}"
MODELDIR="${RUN_DIR}/models"
LOGDIR="${RUN_DIR}/logs"
LOGFILE="${RUN_DIR}/train_full.log"
PIDFILE="${RUN_DIR}/train.pid"
mkdir -p "$MODELDIR" "$LOGDIR"

# stop previous trainers
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

# worker count: match container CPU quota when possible
python3 - <<'PY'
from pathlib import Path
import os
workers = 6
quota = period = None
try:
    quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
    period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
    if quota > 0 and period > 0:
        workers = max(2, min(8, int(quota / period)))
except Exception:
    pass
p = Path("/gemini/code/sftc-rider-dispatch/environments/w_factory_config.py")
text = p.read_text(encoding="utf-8")
import re
text2, n = re.subn(
    r'("num_parallel_workers"\s*:\s*)\d+',
    rf'\g<1>{workers}',
    text,
    count=1,
)
if n:
    p.write_text(text2, encoding="utf-8")
print(f"workers -> {workers} (cpu_quota={quota}, period={period})")
PY

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT"
export PYTHONUNBUFFERED=1

echo "starting training at $(date)" | tee -a "$LOGFILE"
echo "RUN_DIR=$RUN_DIR" | tee -a "$LOGFILE"

nohup /root/miniconda3/bin/python -u mappo/ppo_marl_train.py \
  --scenario delivery \
  --candidate-source upstream \
  --models-dir "$MODELDIR" \
  --logs-dir "$LOGDIR" \
  >> "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "TRAIN_PID=$PID"
echo "MODELS=$MODELDIR"
echo "LOGS=$LOGDIR"
echo "TRAIN_LOG=$LOGFILE"
# stable symlink for local backup / supervisor
ln -sfn "$RUN_DIR" "$CODE_ROOT/runs/latest"
sleep 8
ps -p "$PID" -o pid,etime,cmd --no-headers || echo "MAIN NOT RUNNING"
pgrep -af 'ppo_marl_train|spawn_main' | head
echo "--- log tail ---"
tail -20 "$LOGFILE"
