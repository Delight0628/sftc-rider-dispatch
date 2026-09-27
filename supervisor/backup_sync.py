"""
Local backup for remote training artifacts.

- Polls SSH every INTERVAL seconds
- Copies: train logs, metrics, episode folders, and any new model files
- Immediately downloads a checkpoint when a new .h5/.keras appears

Usage (PowerShell):
  $env:TRAIN_SSH_PASSWORD = '...'
  python supervisor\backup_sync.py --host direct.virtaicloud.com --port 30022 --user ...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def askpass(password: str) -> str:
    fd, path = tempfile.mkstemp(prefix="askpass_", suffix=".cmd")
    os.close(fd)
    with open(path, "w", encoding="ascii") as f:
        f.write(f"@echo {password}\n")
    return path


def ssh_env(password: str) -> tuple[dict, str]:
    ap = askpass(password)
    env = os.environ.copy()
    env["SSH_ASKPASS"] = ap
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env.setdefault("DISPLAY", "localhost:0")
    return env, ap


def run_scp(cfg: dict, src: str, dst: str, recursive: bool = False) -> bool:
    env, ap = ssh_env(cfg["password"])
    try:
        cmd = [
            "scp",
            "-P", str(cfg["port"]),
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=NUL",
            "-o", "PreferredAuthentications=password",
            "-o", "PubkeyAuthentication=no",
            "-o", "NumberOfPasswordPrompts=1",
            "-o", "ConnectTimeout=10",
        ]
        if recursive:
            cmd.append("-r")
        cmd.extend([src, dst])
        proc = subprocess.run(cmd, capture_output=True, timeout=180, env=env)
        return proc.returncode == 0
    except Exception as e:
        print(f"  scp fail {src}: {e}", flush=True)
        return False
    finally:
        try:
            os.remove(ap)
        except OSError:
            pass


def run_ssh(cfg: dict, remote_cmd: str, timeout: int = 30) -> str:
    env, ap = ssh_env(cfg["password"])
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-p", str(cfg["port"]),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=NUL",
                "-o", "PreferredAuthentications=password",
                "-o", "PubkeyAuthentication=no",
                "-o", "NumberOfPasswordPrompts=1",
                "-o", "ConnectTimeout=10",
                f"{cfg['user']}@{cfg['host']}",
                remote_cmd,
            ],
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        return (proc.stdout or b"").decode("utf-8", errors="replace")
    finally:
        try:
            os.remove(ap)
        except OSError:
            pass


def main() -> int:
    p = argparse.ArgumentParser(description="Backup remote MAPPO artifacts to local")
    p.add_argument("--host", default=os.environ.get("TRAIN_SSH_HOST", "direct.virtaicloud.com"))
    p.add_argument("--port", type=int, default=int(os.environ.get("TRAIN_SSH_PORT", "30022")))
    p.add_argument("--user", default=os.environ.get(
        "TRAIN_SSH_USER",
        "delight@root@ssh-acbb566263aa1fdfbe1f7d9aadd0e920.mczzavdaekmx",
    ))
    p.add_argument("--password", default=os.environ.get("TRAIN_SSH_PASSWORD", ""))
    p.add_argument("--remote-code", default="/gemini/code/sftc-rider-dispatch")
    p.add_argument("--local-dir", default=r"D:\rider-dispatch-mappo\training_backup")
    p.add_argument("--interval", type=float, default=180.0)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if not args.password:
        print("missing password: set TRAIN_SSH_PASSWORD or --password")
        return 2

    cfg = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
    }
    remote_code = args.remote_code.rstrip("/")
    local_root = Path(args.local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    seen_models: set[str] = set()
    marker = local_root / ".seen_models.txt"
    if marker.exists():
        seen_models = set(marker.read_text(encoding="utf-8").splitlines())

    user_at = f"{cfg['user']}@{cfg['host']}"

    def sync_once(tag: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] sync ({tag})", flush=True)
        # resolve latest run
        out = run_ssh(cfg, f"readlink -f {remote_code}/runs/latest 2>/dev/null; ls {remote_code}/runs 2>/dev/null | tail -5")
        run_path = ""
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("/"):
                run_path = line
                break
        if not run_path:
            # fallback: newest run dir
            out2 = run_ssh(cfg, f"ls -d {remote_code}/runs/run_* {remote_code}/runs/*_fulltrain 2>/dev/null | tail -1")
            run_path = out2.strip().splitlines()[-1] if out2.strip() else ""
        if not run_path:
            print("  no remote run dir yet", flush=True)
            return

        run_name = os.path.basename(run_path.rstrip("/"))
        local_run = local_root / run_name
        local_run.mkdir(parents=True, exist_ok=True)
        remote_prefix = f"{user_at}:{run_path}"

        # logs + metrics + episode folders
        for rel in ("train_full.log", "metrics.jsonl", "logs/metrics.jsonl"):
            run_scp(cfg, f"{remote_prefix}/{rel}", str(local_run / Path(rel).name))

        # episode logger root: logs/run_*/ or logs root
        run_scp(cfg, f"{remote_prefix}/logs", str(local_run / "logs"), recursive=True)
        # episodes may live under logs/run_*/episodes
        run_scp(cfg, f"{remote_prefix}/logs/run_*", str(local_run / "logs"), recursive=True)

        # models: list and pull new ones immediately
        models_out = run_ssh(
            cfg,
            f"find {run_path}/models -type f \\( -name '*.h5' -o -name '*.keras' -o -name '*_meta.json' -o -name '*.weights.h5' \\) 2>/dev/null",
        )
        new_files = []
        for line in models_out.splitlines():
            path = line.strip()
            if not path:
                continue
            if path in seen_models:
                continue
            new_files.append(path)

        if new_files:
            print(f"  NEW checkpoints: {len(new_files)}", flush=True)
            for remote_file in new_files:
                rel = remote_file.replace(run_path, "").lstrip("/")
                dest = local_run / "models" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                ok = run_scp(cfg, f"{user_at}:{remote_file}", str(dest))
                if ok:
                    seen_models.add(remote_file)
                    print(f"  saved {dest}", flush=True)
            marker.write_text("\n".join(sorted(seen_models)), encoding="utf-8")
        else:
            print("  no new checkpoints", flush=True)

    # first sync immediately (also pulls any existing artifacts)
    sync_once("init")
    if args.once:
        return 0

    while True:
        time.sleep(max(30.0, args.interval))
        try:
            sync_once("loop")
        except Exception as e:
            print(f"sync error: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
