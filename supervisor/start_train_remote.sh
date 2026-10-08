#!/bin/bash
# Launch Hybrid dispatch training (edge-Q + constrained matching) with artifacts
# under the CODE path (persist across job restarts).
# Layout: models + logs live next to the code (runs/<stamp>_hybrid/).
set -u

CODE_ROOT=/gemini/code/sftc-rider-dispatch
STAMP=$(date +%m%d_%H%M)
RUN_NAME="${STAMP}_hybrid"
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
pkill -f hybrid_train.py 2>/dev/null || true
pkill -f 'spawn_main' 2>/dev/null || true
sleep 1

# worker count: match container CPU quota when possible
WORKERS=$(python3 - <<'PY'
from pathlib import Path
workers = 4
try:
    quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text().strip())
    period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
    if quota > 0 and period > 0:
        workers = max(1, min(6, int(quota / period)))
except Exception:
    pass
print(workers)
PY
)
echo "workers -> ${WORKERS}"

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT"
export PYTHONUNBUFFERED=1

echo "starting training at $(date)" | tee -a "$LOGFILE"
echo "RUN_DIR=$RUN_DIR" | tee -a "$LOGFILE"

# 真实果洛样本训练（业务向）：订单/骑手 xlsx + episode 窗口 40 单
nohup /root/miniconda3/bin/python -u hybrid_train.py \
  --real-orders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --real-riders "骑手派单仿真样本_果洛藏族自治州_20260915_全量.xlsx" \
  --episode-order-size 40 \
  --num-parallel-workers "$WORKERS" \
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
pgrep -af 'hybrid_train|spawn_main' | head
echo "--- log tail ---"
tail -20 "$LOGFILE"
