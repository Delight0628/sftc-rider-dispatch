"""
Local real-time training supervisor dashboard.

Run on your Windows PC:
  python supervisor/local_dashboard.py

Then open http://127.0.0.1:8787

It SSHs into the training box, pulls metrics.jsonl + host stats,
and renders RL training curves live in the browser.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# SSH config (override via CLI flags or env: TRAIN_SSH_PASSWORD)
# ---------------------------------------------------------------------------
DEFAULT_SSH = {
    "host": os.environ.get("TRAIN_SSH_HOST", "direct.virtaicloud.com"),
    "port": int(os.environ.get("TRAIN_SSH_PORT", "30022")),
    "user": os.environ.get(
        "TRAIN_SSH_USER",
        "delight@root@ssh-acbb566263aa1fdfbe1f7d9aadd0e920.mczzavdaekmx",
    ),
    "password": os.environ.get("TRAIN_SSH_PASSWORD", ""),
    "remote_metrics": os.environ.get(
        "TRAIN_METRICS_PATH",
        "/gemini/code/sftc-rider-dispatch/runs/latest/logs/metrics.jsonl",
    ),
    "remote_log": os.environ.get(
        "TRAIN_LOG_PATH",
        "/gemini/code/sftc-rider-dispatch/runs/latest/train_full.log",
    ),
}

STATE: Dict[str, Any] = {
    "episodes": [],
    "events": [],
    "host": {},
    "status": {"connected": False, "last_poll": 0, "error": "", "message": "初始化中…"},
    "ssh": dict(DEFAULT_SSH),
}


def _askpass_path(password: str) -> str:
    fd, path = tempfile.mkstemp(prefix="askpass_", suffix=".cmd")
    os.close(fd)
    with open(path, "w", encoding="ascii") as f:
        f.write(f"@echo {password}\n")
    return path


def _ssh_base(cfg: Dict[str, Any]) -> List[str]:
    return [
        "ssh",
        "-p", str(cfg.get("port", 22)),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "PreferredAuthentications=password",
        "-o", "PubkeyAuthentication=no",
        "-o", "NumberOfPasswordPrompts=1",
        "-o", "ConnectTimeout=8",
        f"{cfg['user']}@{cfg['host']}",
    ]


def _run_ssh(cfg: Dict[str, Any], remote_cmd: str, timeout: int = 20) -> str:
    askpass = _askpass_path(cfg.get("password", ""))
    env = os.environ.copy()
    env["SSH_ASKPASS"] = askpass
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env.setdefault("DISPLAY", "localhost:0")
    try:
        proc = subprocess.run(
            _ssh_base(cfg) + [remote_cmd],
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        out = proc.stdout or b""
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(err or f"ssh exit {proc.returncode}")
        return out.decode("utf-8", errors="replace")
    finally:
        try:
            os.remove(askpass)
        except OSError:
            pass


def _parse_metrics_blob(blob: str, limit: int = 800) -> tuple[List[dict], List[dict]]:
    episodes: List[dict] = []
    events: List[dict] = []
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "event":
            events.append(obj)
        else:
            obj.setdefault("type", "episode")
            episodes.append(obj)
    return episodes[-limit:], events[-50:]


def _parse_host_stats(text: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return stats


def poll_once(cfg: Dict[str, Any]) -> None:
    remote_metrics = cfg.get("remote_metrics") or DEFAULT_SSH["remote_metrics"]
    remote_log = cfg.get("remote_log") or DEFAULT_SSH["remote_log"]
    host_cmd = (
        "echo __HOST__; "
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader,nounits 2>/dev/null | head -1; "
        "echo __CPU__; "
        "q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); "
        "p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 1); "
        "if [ \"$q\" -gt 0 ] 2>/dev/null; then echo $((q/p)); else nproc; fi; "
        "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-; "
        "awk '{printf \"%.0f\\n\", $1/1024/1024}' /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null; "
        "awk '{printf \"%.0f\\n\", $1/1024/1024}' /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null; "
        "echo __PROC__; "
        "ps aux | grep -E 'ppo_marl_train|spawn_main' | grep -v grep | wc -l; "
        "echo __LOG__; "
        f"wc -l < {remote_log}; "
        f"tail -3 {remote_log}; "
        "echo __METRICS__; "
        f"cat {remote_metrics}"
    )
    try:
        out = _run_ssh(cfg, host_cmd, timeout=25)
        STATE["status"]["connected"] = True
        STATE["status"]["error"] = ""
        STATE["status"]["last_poll"] = time.time()

        host: Dict[str, Any] = {"gpu_raw": "", "cpu_count": 0, "cpu_model": "",
                                "mem_total_mb": 0, "mem_used_mb": 0, "mem_avail_mb": 0,
                                "proc_count": 0, "log_lines": 0, "log_tail": ""}
        metrics_blob = ""
        section = ""
        for line in out.splitlines():
            s = line.strip()
            if s == "__HOST__":
                section = "host"
                continue
            if s == "__CPU__":
                section = "cpu"
                continue
            if s == "__PROC__":
                section = "proc"
                continue
            if s == "__LOG__":
                section = "log"
                continue
            if s == "__METRICS__":
                section = "metrics"
                continue
            if section == "host":
                if s:
                    host["gpu_raw"] = s
            elif section == "cpu":
                if s.isdigit() and host["cpu_count"] == 0:
                    host["cpu_count"] = int(s)
                elif ("intel" in s.lower() or "amd" in s.lower() or "xeon" in s.lower()) and not host["cpu_model"]:
                    host["cpu_model"] = s
                elif s:
                    # cgroup: limit_mb then usage_mb (floats)
                    try:
                        val = float(s)
                    except ValueError:
                        parts = s.split()
                        if len(parts) >= 3:
                            try:
                                host["mem_total_mb"] = int(float(parts[0]))
                                host["mem_used_mb"] = int(float(parts[1]))
                                host["mem_avail_mb"] = int(float(parts[2]))
                            except ValueError:
                                pass
                    else:
                        if host["mem_total_mb"] == 0:
                            host["mem_total_mb"] = int(val)
                        elif host["mem_used_mb"] == 0:
                            host["mem_used_mb"] = int(val)
            elif section == "proc":
                if s.isdigit():
                    host["proc_count"] = int(s)
            elif section == "log":
                if s.isdigit() and host["log_lines"] == 0:
                    host["log_lines"] = int(s)
                elif s:
                    host["log_tail"] = (host["log_tail"] + "\n" + s).strip()
            elif section == "metrics":
                metrics_blob += line + "\n"

        # GPU parse: name, util, mem_used, mem_total
        if host.get("gpu_raw"):
            parts = [p.strip() for p in host["gpu_raw"].split(",")]
            if len(parts) >= 4:
                host["gpu_name"] = parts[0]
                try:
                    host["gpu_util"] = float(parts[1])
                    host["gpu_mem_used"] = float(parts[2])
                    host["gpu_mem_total"] = float(parts[3])
                except ValueError:
                    pass

        episodes, events = _parse_metrics_blob(metrics_blob)
        STATE["episodes"] = episodes
        STATE["events"] = events
        STATE["host"] = host

        if episodes:
            last = episodes[-1]
            STATE["status"]["message"] = (
                f"回合 {last.get('episode', '?')}/{last.get('max_episodes', '?')} "
                f"| reward {last.get('episode_reward', 0):.1f} "
                f"| score {last.get('score', 0):.3f}"
            )
        else:
            STATE["status"]["message"] = "已连接，等待首个回合指标…"
    except Exception as e:
        STATE["status"]["connected"] = False
        STATE["status"]["error"] = str(e)
        STATE["status"]["message"] = f"SSH 轮询失败: {e}"


def poll_loop(interval: float) -> None:
    while True:
        try:
            poll_once(STATE["ssh"])
        except Exception as e:
            STATE["status"]["error"] = str(e)
        time.sleep(interval)


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>MAPPO 训练监督器</title>
<style>
:root{
  --bg:#0f1419; --panel:#1a2332; --ink:#e7ecf3; --muted:#8b9bb4;
  --accent:#3d9cf0; --good:#3dd68c; --warn:#f0b429; --bad:#f07178;
  --grid:#243044; --line:#2a3548;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.45 -apple-system,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif}
header{display:flex;align-items:center;gap:16px;padding:14px 20px;
  border-bottom:1px solid var(--line);background:linear-gradient(180deg,#152033,#0f1419)}
h1{margin:0;font-size:18px;font-weight:600;letter-spacing:.02em}
.badge{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600}
.badge.on{background:rgba(61,214,140,.15);color:var(--good)}
.badge.off{background:rgba(240,113,120,.15);color:var(--bad)}
.msg{color:var(--muted);margin-left:auto;font-size:12px}
main{padding:16px 20px 32px;max-width:1400px;margin:0 auto}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:22px;font-weight:650;margin-top:4px;font-variant-numeric:tabular-nums}
.card .s{color:var(--muted);font-size:11px;margin-top:2px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:12px}
.panel h2{margin:0 0 8px;font-size:13px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.05em}
canvas{width:100%;height:220px;display:block}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;
  font-variant-numeric:tabular-nums}
th{color:var(--muted);font-weight:500;text-align:right}
th:first-child,td:first-child{text-align:left}
tr:hover td{background:rgba(61,156,240,.06)}
.log{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--muted);
  white-space:pre-wrap;max-height:120px;overflow:auto}
@media (max-width:900px){.grid2{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>🛵 MAPPO 训练监督器</h1>
  <span id="conn" class="badge off">连接中…</span>
  <span id="msg" class="msg">…</span>
</header>
<main>
  <div class="cards" id="cards"></div>

  <div class="grid2">
    <div class="panel"><h2>Episode Reward</h2><canvas id="c_reward"></canvas></div>
    <div class="panel"><h2>综合评分 Score / 完成率</h2><canvas id="c_score"></canvas></div>
    <div class="panel"><h2>Policy Loss（Actor / Critic）</h2><canvas id="c_loss"></canvas></div>
    <div class="panel"><h2>Entropy / KL / Clip Fraction</h2><canvas id="c_entropy"></canvas></div>
    <div class="panel"><h2>KPI：延期 / 利用率 / Makespan</h2><canvas id="c_kpi"></canvas></div>
    <div class="panel"><h2>吞吐：回合耗时 / 采样更新</h2><canvas id="c_perf"></canvas></div>
  </div>

  <div class="panel">
    <h2>最近回合</h2>
    <div style="overflow:auto;max-height:320px">
      <table id="tbl">
        <thead><tr>
          <th>Ep</th><th>Phase</th><th>Reward</th><th>Score</th><th>完成%</th>
          <th>Actor</th><th>Critic</th><th>Ent</th><th>KL</th><th>Clip</th>
          <th>LR</th><th>延期</th><th>耗时</th>
        </tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h2>主机 / 日志</h2>
    <div id="host" style="color:var(--muted);margin-bottom:8px"></div>
    <div class="log" id="log"></div>
  </div>
</main>
<script>
function series(eps, key){
  return eps.map(e => ({x: e.episode, y: Number(e[key] ?? 0)}));
}
function draw(id, lines, opts={}){
  const cv = document.getElementById(id);
  if(!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w*dpr; cv.height = h*dpr;
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  const pad = {l:44,r:10,t:12,b:24};
  const pw = w-pad.l-pad.r, ph = h-pad.t-pad.b;
  const all = lines.flatMap(l => l.data);
  if(!all.length){
    ctx.fillStyle='#8b9bb4'; ctx.font='12px sans-serif';
    ctx.fillText('等待数据…', pad.l, h/2); return;
  }
  let xmin=Math.min(...all.map(p=>p.x)), xmax=Math.max(...all.map(p=>p.x));
  let ymin=Math.min(...all.map(p=>p.y)), ymax=Math.max(...all.map(p=>p.y));
  if(xmax===xmin) xmax=xmin+1;
  if(ymax===ymin){ ymax=ymin+1; ymin=ymin-1; }
  const ypad=(ymax-ymin)*0.08; ymin-=ypad; ymax+=ypad;
  if(opts.zeroBase && ymin>0) ymin=0;
  const X=x=>pad.l+(x-xmin)/(xmax-xmin)*pw;
  const Y=y=>pad.t+ph-(y-ymin)/(ymax-ymin)*ph;
  // grid
  ctx.strokeStyle='#243044'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad.t+ph*i/4;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+pw,y); ctx.stroke();
    const val=ymax-(ymax-ymin)*i/4;
    ctx.fillStyle='#8b9bb4'; ctx.font='10px sans-serif'; ctx.textAlign='right';
    ctx.fillText(fmt(val), pad.l-6, y+3);
  }
  // lines
  for(const ln of lines){
    if(!ln.data.length) continue;
    ctx.strokeStyle=ln.color; ctx.lineWidth=1.8; ctx.beginPath();
    ln.data.forEach((p,i)=>{ const x=X(p.x),y=Y(p.y); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
    ctx.stroke();
    // last point
    const last=ln.data[ln.data.length-1];
    ctx.fillStyle=ln.color; ctx.beginPath();
    ctx.arc(X(last.x),Y(last.y),3,0,Math.PI*2); ctx.fill();
  }
  // legend
  let lx=pad.l+8;
  ctx.font='11px sans-serif'; ctx.textAlign='left';
  for(const ln of lines){
    ctx.fillStyle=ln.color; ctx.fillRect(lx, pad.t+2, 10, 3);
    ctx.fillStyle='#8b9bb4'; ctx.fillText(ln.name, lx+14, pad.t+7);
    lx += 14 + ctx.measureText(ln.name).width + 16;
  }
}
function fmt(v){
  if(Math.abs(v)>=1000) return (v/1000).toFixed(1)+'k';
  if(Math.abs(v)>=10) return v.toFixed(1);
  return v.toFixed(3);
}
function render(data){
  const eps = data.episodes || [];
  const host = data.host || {};
  const st = data.status || {};
  const conn = document.getElementById('conn');
  conn.textContent = st.connected ? '已连接' : '未连接';
  conn.className = 'badge ' + (st.connected ? 'on' : 'off');
  document.getElementById('msg').textContent = st.message || '';

  const last = eps[eps.length-1] || {};
  const cards = [
    ['回合', `${last.episode ?? '-'} / ${last.max_episodes ?? '-'}`, st.message || ''],
    ['Episode Reward', fmt(last.episode_reward ?? 0), `avg worker ${fmt(last.avg_worker_reward ?? 0)}`],
    ['Score', fmt(last.score ?? 0), `best ${fmt(last.best_score ?? 0)}`],
    ['完成率', fmt(last.completion_rate ?? 0)+'%', `完成 ${fmt(last.completed ?? 0)}`],
    ['Actor Loss', fmt(last.actor_loss ?? 0), `clip ${fmt(last.clip_fraction ?? 0)}`],
    ['Critic Loss', fmt(last.critic_loss ?? 0), `kl ${fmt(last.approx_kl ?? 0)}`],
    ['Entropy', fmt(last.entropy ?? 0), `ent_coef ${fmt(last.entropy_coeff ?? 0)}`],
    ['LR', fmt(last.learning_rate ?? 0), `phase ${last.phase || '-'}`],
    ['回合耗时', (last.iteration_duration ?? 0).toFixed(1)+'s',
      `采样 ${(last.collect_duration??0).toFixed(1)}s / 更新 ${(last.update_duration??0).toFixed(1)}s`],
    ['延期', fmt(last.tardiness ?? 0), `makespan ${fmt(last.makespan ?? 0)}`],
    ['利用率', fmt((last.utilization ?? 0)*100)+'%', `workers ${last.workers ?? '-'}`],
    ['GPU', host.gpu_util!=null ? host.gpu_util+'%' : '-', host.gpu_name || 'n/a'],
    ['进程', host.proc_count ?? '-', 'train workers'],
    ['内存', host.mem_used_mb ? (host.mem_used_mb/1024).toFixed(1)+'G' : '-',
      host.mem_total_mb ? `/${(host.mem_total_mb/1024).toFixed(0)}G` : ''],
  ];
  document.getElementById('cards').innerHTML = cards.map(c=>`
    <div class="card"><div class="k">${c[0]}</div>
    <div class="v">${c[1]}</div><div class="s">${c[2]||''}</div></div>`).join('');

  draw('c_reward', [
    {name:'reward', color:'#3d9cf0', data:series(eps,'episode_reward')},
    {name:'kpi_reward', color:'#3dd68c', data:series(eps,'kpi_reward')},
  ]);
  draw('c_score', [
    {name:'score', color:'#f0b429', data:series(eps,'score')},
    {name:'best', color:'#c792ea', data:series(eps,'best_score')},
    {name:'completion%', color:'#3dd68c', data:series(eps,'completion_rate')},
  ]);
  draw('c_loss', [
    {name:'actor', color:'#f07178', data:series(eps,'actor_loss')},
    {name:'critic', color:'#3d9cf0', data:series(eps,'critic_loss')},
    {name:'bc', color:'#8b9bb4', data:series(eps,'bc_loss')},
  ]);
  draw('c_entropy', [
    {name:'entropy', color:'#c792ea', data:series(eps,'entropy')},
    {name:'approx_kl', color:'#f0b429', data:series(eps,'approx_kl')},
    {name:'clip_frac', color:'#3dd68c', data:series(eps,'clip_fraction')},
  ]);
  draw('c_kpi', [
    {name:'tardiness', color:'#f07178', data:series(eps,'tardiness')},
    {name:'util*100', color:'#3d9cf0', data:series(eps,'utilization').map(p=>({x:p.x,y:p.y*100}))},
    {name:'makespan', color:'#8b9bb4', data:series(eps,'makespan')},
  ]);
  draw('c_perf', [
    {name:'iter_s', color:'#3d9cf0', data:series(eps,'iteration_duration')},
    {name:'collect_s', color:'#3dd68c', data:series(eps,'collect_duration')},
    {name:'update_s', color:'#f0b429', data:series(eps,'update_duration')},
  ]);

  const rows = eps.slice(-30).reverse().map(e=>`<tr>
    <td>${e.episode}</td>
    <td>${e.phase||''}</td>
    <td>${fmt(e.episode_reward)}</td>
    <td>${fmt(e.score)}</td>
    <td>${fmt(e.completion_rate)}%</td>
    <td>${fmt(e.actor_loss)}</td>
    <td>${fmt(e.critic_loss)}</td>
    <td>${fmt(e.entropy)}</td>
    <td>${fmt(e.approx_kl)}</td>
    <td>${fmt(e.clip_fraction)}</td>
    <td>${fmt(e.learning_rate)}</td>
    <td>${fmt(e.tardiness)}</td>
    <td>${(e.iteration_duration||0).toFixed(1)}s</td>
  </tr>`).join('');
  document.querySelector('#tbl tbody').innerHTML = rows;

  document.getElementById('host').textContent =
    `GPU: ${host.gpu_name||'n/a'} ${host.gpu_util!=null?host.gpu_util+'%':''} | ` +
    `CPU: ${host.cpu_count||'?'} 核 ${host.cpu_model||''} | ` +
    `内存: ${host.mem_used_mb||0}/${host.mem_total_mb||0} MB | ` +
    `进程: ${host.proc_count||0} | 日志行: ${host.log_lines||0}`;
  document.getElementById('log').textContent = host.log_tail || data.status?.error || '';
}

async function tick(){
  try{
    const r = await fetch('/api/state');
    const data = await r.json();
    render(data);
  }catch(e){
    document.getElementById('msg').textContent = '刷新失败: '+e;
  }
}
tick();
setInterval(tick, 2500);
window.addEventListener('resize', tick);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path.startswith("/api/state"):
            payload = {
                "episodes": STATE["episodes"],
                "events": STATE["events"],
                "host": STATE["host"],
                "status": STATE["status"],
            }
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            return
        if self.path.startswith("/api/refresh"):
            poll_once(STATE["ssh"])
            self._send(200, json.dumps({"ok": True, "status": STATE["status"]}, ensure_ascii=False).encode("utf-8"))
            return
        self._send(404, b'{"error":"not found"}')


def main():
    global DEFAULT_SSH
    p = argparse.ArgumentParser(description="MAPPO local training supervisor")
    p.add_argument("--host", default=DEFAULT_SSH["host"])
    p.add_argument("--port", type=int, default=DEFAULT_SSH["port"])
    p.add_argument("--user", default=DEFAULT_SSH["user"])
    p.add_argument("--password", default=os.environ.get("TRAIN_SSH_PASSWORD", ""))
    p.add_argument("--remote-metrics", default=DEFAULT_SSH["remote_metrics"])
    p.add_argument("--remote-log", default=DEFAULT_SSH["remote_log"])
    p.add_argument("--listen", type=int, default=8787)
    p.add_argument("--interval", type=float, default=3.0)
    args = p.parse_args()

    STATE["ssh"] = {
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "password": args.password,
        "remote_metrics": args.remote_metrics,
        "remote_log": args.remote_log,
    }

    t = threading.Thread(target=poll_loop, args=(args.interval,), daemon=True)
    t.start()

    # warm-up
    poll_once(STATE["ssh"])

    httpd = ThreadingHTTPServer(("127.0.0.1", args.listen), Handler)
    print(f"✅ 监督器已启动: http://127.0.0.1:{args.listen}")
    print(f"   远程: {args.user}@{args.host}:{args.port}")
    print(f"   指标: {args.remote_metrics}")
    print("   按 Ctrl+C 退出")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
