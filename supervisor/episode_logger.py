"""
Per-episode complete log writer (aligned with MARL_FOR_W_Factory style).

Directory layout under the run logs root:
  <logs_root>/
    run_<timestamp>/
      train.log                 # full training stdout (rotated externally)
      metrics.jsonl             # one JSON per episode
      episodes/
        ep_000001/
          episode.log           # human-readable complete episode log
          metrics.json          # structured metrics for this episode
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class EpisodeLogger:
    """Write a complete log folder for every training episode."""

    def __init__(self, logs_root: str, run_name: Optional[str] = None):
        self.logs_root = logs_root
        stamp = run_name or time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(logs_root, f"run_{stamp}")
        self.episodes_dir = os.path.join(self.run_dir, "episodes")
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        os.makedirs(self.episodes_dir, exist_ok=True)
        # touch metrics
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "a", encoding="utf-8"):
                pass
        self._episode_index = 0

    def episode_folder(self, episode_1based: int) -> str:
        path = os.path.join(self.episodes_dir, f"ep_{episode_1based:06d}")
        os.makedirs(path, exist_ok=True)
        return path

    def save_episode(
        self,
        *,
        episode_1based: int,
        console_lines: List[str],
        metrics: Dict[str, Any],
        extra_notes: Optional[List[str]] = None,
    ) -> str:
        """Save full episode log + metrics. Returns episode folder path."""
        self._episode_index = max(self._episode_index, episode_1based)
        folder = self.episode_folder(episode_1based)

        # 1) human-readable complete log
        log_path = os.path.join(folder, "episode.log")
        header = [
            "=" * 72,
            f"训练回合 {episode_1based}",
            f"时间: {metrics.get('wall_time') or time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"阶段: {metrics.get('phase', '-')} | 任务: {metrics.get('episode_task', '-')}",
            "=" * 72,
        ]
        body = list(console_lines or [])
        if extra_notes:
            body.append("")
            body.extend(extra_notes)
        body.append("")
        body.append("--- 结构化指标 ---")
        for key in (
            "episode_reward",
            "avg_worker_reward",
            "actor_loss",
            "critic_loss",
            "entropy",
            "approx_kl",
            "clip_fraction",
            "bc_loss",
            "bc_coeff",
            "learning_rate",
            "entropy_coeff",
            "score",
            "best_score",
            "completion_rate",
            "makespan",
            "completed",
            "utilization",
            "tardiness",
            "kpi_reward",
            "iteration_duration",
            "collect_duration",
            "update_duration",
            "total_steps",
            "workers",
            "foundation_completed",
            "generalization_active",
            "target_achieved_count",
        ):
            if key in metrics:
                body.append(f"{key}: {metrics[key]}")

        content = "\n".join(header + body) + "\n"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 2) structured metrics for this episode
        json_path = os.path.join(folder, "metrics.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)

        # 3) append to run-level jsonl
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False, default=float) + "\n")
            f.flush()

        # 4) convenience index
        self._append_index(episode_1based, metrics, folder)
        return folder

    def _append_index(self, episode_1based: int, metrics: Dict[str, Any], folder: str) -> None:
        index_path = os.path.join(self.episodes_dir, "index.csv")
        write_header = not os.path.exists(index_path)
        cols = [
            "episode",
            "wall_time",
            "phase",
            "episode_reward",
            "score",
            "completion_rate",
            "actor_loss",
            "critic_loss",
            "entropy",
            "approx_kl",
            "tardiness",
            "makespan",
            "utilization",
            "folder",
        ]
        with open(index_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(cols) + "\n")
            row = [
                str(episode_1based),
                str(metrics.get("wall_time", "")),
                str(metrics.get("phase", "")),
                str(metrics.get("episode_reward", "")),
                str(metrics.get("score", "")),
                str(metrics.get("completion_rate", "")),
                str(metrics.get("actor_loss", "")),
                str(metrics.get("critic_loss", "")),
                str(metrics.get("entropy", "")),
                str(metrics.get("approx_kl", "")),
                str(metrics.get("tardiness", "")),
                str(metrics.get("makespan", "")),
                str(metrics.get("utilization", "")),
                folder.replace("\\", "/"),
            ]
            f.write(",".join(row) + "\n")

    def write_event(self, kind: str, **fields: Any) -> None:
        payload = {
            "type": "event",
            "event": kind,
            "ts": time.time(),
            "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        payload.update(fields)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=float) + "\n")
            f.flush()

    def summary_path(self) -> str:
        return os.path.join(self.run_dir, "summary.json")

    def write_summary(self, data: Dict[str, Any]) -> None:
        with open(self.summary_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=float)
