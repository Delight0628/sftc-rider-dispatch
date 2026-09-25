"""Append-only JSONL metrics recorder for MAPPO training episodes."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional


class MetricsRecorder:
    """Write one JSON object per training episode to a .jsonl file."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._start_ts = time.time()
        # touch file so dashboard can tail immediately
        if not os.path.exists(self.path):
            with open(self.path, "a", encoding="utf-8"):
                pass

    def write_episode(self, record: Dict[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("ts", time.time())
        payload.setdefault("wall_time", time.strftime("%Y-%m-%d %H:%M:%S"))
        line = json.dumps(payload, ensure_ascii=False, default=float)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_event(self, kind: str, **fields: Any) -> None:
        payload = {
            "type": "event",
            "event": kind,
            "ts": time.time(),
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        payload.update(fields)
        line = json.dumps(payload, ensure_ascii=False, default=float)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


def build_episode_record(
    *,
    episode: int,
    max_episodes: int,
    phase: str,
    episode_reward: float,
    losses: Dict[str, float],
    kpi: Dict[str, float],
    score: float,
    completion_rate: float,
    iteration_duration: float,
    collect_duration: float,
    update_duration: float,
    total_steps: int,
    learning_rate: float,
    entropy_coeff: float,
    best_score: Optional[float] = None,
    target_achieved_count: int = 0,
    foundation_completed: bool = False,
    generalization_active: bool = False,
    episode_task: str = "",
    workers: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    steps_per_sec = 0.0
    if iteration_duration and iteration_duration > 0:
        # approximate env steps/sec across workers
        steps_per_sec = float(workers or 1) * float(kpi.get("mean_steps", 0) or 0) / iteration_duration

    record = {
        "type": "episode",
        "episode": int(episode),
        "max_episodes": int(max_episodes),
        "progress_pct": round(100.0 * (episode + 1) / max(max_episodes, 1), 2),
        "phase": phase,
        "episode_task": episode_task,
        # RL core
        "episode_reward": float(episode_reward),
        "actor_loss": float(losses.get("actor_loss", 0.0)),
        "critic_loss": float(losses.get("critic_loss", 0.0)),
        "entropy": float(losses.get("entropy", 0.0)),
        "approx_kl": float(losses.get("approx_kl", 0.0)),
        "clip_fraction": float(losses.get("clip_fraction", 0.0)),
        "bc_loss": float(losses.get("bc_loss", 0.0)),
        "bc_coeff": float(losses.get("bc_coeff", 0.0)),
        "learning_rate": float(learning_rate),
        "entropy_coeff": float(entropy_coeff),
        # KPI / task
        "score": float(score),
        "completion_rate": float(completion_rate),
        "makespan": float(kpi.get("mean_makespan", 0.0)),
        "completed": float(kpi.get("mean_completed_parts", 0.0)),
        "utilization": float(kpi.get("mean_utilization", 0.0)),
        "tardiness": float(kpi.get("mean_tardiness", 0.0)),
        "kpi_reward": float(kpi.get("mean_reward", 0.0)),
        # throughput
        "iteration_duration": float(iteration_duration),
        "collect_duration": float(collect_duration),
        "update_duration": float(update_duration),
        "steps_per_sec": float(steps_per_sec),
        "total_steps": int(total_steps),
        "workers": int(workers),
        # training state
        "best_score": float(best_score) if best_score is not None else float(score),
        "target_achieved_count": int(target_achieved_count),
        "foundation_completed": bool(foundation_completed),
        "generalization_active": bool(generalization_active),
    }
    if extra:
        record.update(extra)
    return record
