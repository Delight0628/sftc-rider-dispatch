import os
import sys
import time
import datetime
import argparse
import subprocess
from pathlib import Path
import signal
import shutil
import threading

# 全局变量，用于存储需要监控的子进程
child_processes = []

def cleanup(signum, frame):
    """信号处理函数，用于在脚本退出前清理子进程。"""
    print(f"\n🚦 捕获到信号 {signum}。正在清理后台训练进程...", flush=True)
    for p in child_processes:
        if p.poll() is None:  # 检查进程是否仍在运行
            try:
                # 强制杀死子进程的整个进程组，确保完全终止
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                print(f"🔪 已发送 SIGKILL 到 PID 为 {p.pid} 的进程组。", flush=True)
            except ProcessLookupError:
                pass  # 进程可能已经结束
    sys.exit(0)

# 模型的基础目录，相对于脚本位置
MODELS_BASE_DIR = "mappo/ppo_models"

def find_new_model_dir(base_dir, dirs_before, timeout=120):
    """等待并返回在指定基础目录中新创建的目录路径。"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            current_dirs = set(os.listdir(base_dir))
            new_dirs = current_dirs - dirs_before
            if new_dirs:
                new_dir_name = new_dirs.pop()
                print(f"✅ 成功找到新的模型目录: {new_dir_name}", flush=True)
                return os.path.join(base_dir, new_dir_name)
        except FileNotFoundError:
            # 如果是第一次运行，基础模型目录可能还不存在
            pass
        time.sleep(5)
    print(f"❌ 等待新模型目录超时（{timeout}秒）。", flush=True)
    return None

def run_detached_command(command):
    """在后台运行一个完全分离的命令。"""
    print(f"🚀 正在执行命令:\n   {command}", flush=True)
    subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)

def launch_and_monitor_child(cmd_list, log_file, cwd: str = None):
    """
    启动一个需要被监控的子进程（例如训练脚本），并将其记录下来以便后续清理。
    """
    print(f"🔥 正在启动受监控的训练进程... 日志文件: {log_file}", flush=True)
    with open(log_file, 'wb') as f:
        # start_new_session=True 使子进程成为新会话的领导者，
        # 这使其能抵抗SIGHUP信号（类似于nohup），并创建一个新的进程组。
        p = subprocess.Popen(cmd_list, stdout=f, stderr=f, start_new_session=True, cwd=cwd)
    child_processes.append(p)
    print(f"   -> 训练进程已启动，PID: {p.pid}", flush=True)

def start_log_parser_watcher(log_file_path: str, cwd: str = None, poll_interval_s: int = 15):
    last_mtime = None
    last_size = None
    last_run_ts = 0.0

    base_dir = cwd or os.getcwd()
    try:
        base_dir = os.path.abspath(base_dir)
    except (OSError, TypeError) as e:
        print(f"Warning: Failed to resolve absolute path for base_dir: {e}", flush=True)
    script_path = os.path.join(base_dir, "log_parser.py")
    try:
        script_path = os.path.abspath(script_path)
    except (OSError, TypeError) as e:
        print(f"Warning: Failed to resolve absolute path for script_path: {e}", flush=True)

    def _worker():
        nonlocal last_mtime, last_size, last_run_ts
        while True:
            try:
                if not os.path.exists(log_file_path):
                    time.sleep(poll_interval_s)
                    continue
                st = os.stat(log_file_path)
                mtime = float(st.st_mtime)
                size = int(st.st_size)
                changed = (last_mtime is None) or (mtime != last_mtime) or (size != last_size)
                last_mtime, last_size = mtime, size
                if changed:
                    now_ts = time.time()
                    if now_ts - last_run_ts >= max(5, poll_interval_s):
                        last_run_ts = now_ts
                        try:
                            abs_log_file_path = log_file_path
                            try:
                                abs_log_file_path = os.path.abspath(abs_log_file_path)
                            except (OSError, TypeError) as e:
                                print(f"Warning: Failed to resolve absolute path for log file: {e}", flush=True)
                            r = subprocess.run(
                                [sys.executable, script_path, abs_log_file_path],
                                cwd=base_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                check=False,
                                text=True,
                                encoding='utf-8',
                                errors='ignore'
                            )
                            if r.returncode != 0:
                                print(f"🔴 log_parser 执行失败(返回码={r.returncode})，日志: {abs_log_file_path}", flush=True)
                                if r.stdout:
                                    print(r.stdout[-2000:], flush=True)
                                if r.stderr:
                                    print(r.stderr[-2000:], flush=True)
                        except Exception:
                            pass
                time.sleep(poll_interval_s)
            except Exception:
                time.sleep(poll_interval_s)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

def monitor_and_launch(model_run_dir, main_dir_abs, folder_name, timeout_hours=24, scenario="factory"):
    """
    监控模型目录，并为每个新生成的模型启动评估和调试脚本。
    """
    print(f"👀 开始监控目录: {model_run_dir}", flush=True)

    # 创建用于存放日志和结果的子目录
    debug_dir = os.path.join(main_dir_abs, "debug_marl_behavior")
    eval_dir = os.path.join(main_dir_abs, "evaluation")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    processed_models = set()
    start_time = time.time()
    timeout_seconds = timeout_hours * 3600

    print("🕒 监控循环已启动，将为每个新模型自动触发评估与调试...", flush=True)

    def launch_detached_python(cmd_list, log_path, cwd=None):
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'wb') as f:
                p = subprocess.Popen(cmd_list, stdout=f, stderr=f, start_new_session=True, cwd=cwd)
            child_processes.append(p)
            return p
        except Exception as e:
            print(f"🔴 启动后台任务失败: {e}", flush=True)
            return None

    while time.time() - start_time < timeout_seconds:
        try:
            # 🔧 新增：递归查找所有时间戳子目录中的模型文件
            all_models = {}  # 改为字典，存储 {model_file: full_path}
            
            # 首先检查是否有时间戳子目录（新结构）
            has_timestamp_subdirs = False
            for item in os.listdir(model_run_dir):
                item_path = os.path.join(model_run_dir, item)
                if os.path.isdir(item_path) and item.count('_') == 1 and len(item) == 9:  # 匹配 MMDD_HHMM 格式
                    has_timestamp_subdirs = True
                    # 在时间戳子目录中查找模型
                    for file in os.listdir(item_path):
                        if file.endswith('_actor.keras'):
                            all_models[file] = os.path.join(item_path, file)
            
            # 如果没有时间戳子目录，使用旧逻辑（向后兼容）
            if not has_timestamp_subdirs:
                for file in os.listdir(model_run_dir):
                    if file.endswith('_actor.keras'):
                        all_models[file] = os.path.join(model_run_dir, file)
            
            new_models = set(all_models.keys()) - processed_models

            if not new_models:
                time.sleep(30) # 如果没有新模型，等待30秒
                continue

            for model_file in sorted(list(new_models)): # 按名称排序以保证顺序
                print("\n" + "="*60, flush=True)
                print(f"⭐ 发现新模型: {model_file}", flush=True)
                
                model_path = all_models[model_file]  # 🔧 使用完整路径
                base_name = model_file.replace('.keras', '')

                # 非工厂场景：evaluation.py / debug_marl_behavior.py 目前仅支持工厂场景，
                # 强行启动会产出错误口径的对比数据，故跳过并明确提示（配送场景评估在下一阶段接入）
                if str(scenario).lower() != "factory":
                    print(f"ℹ️ 当前场景={scenario}：跳过工厂专用的 evaluation.py / debug_marl_behavior.py，"
                          f"仅保留训练产物。", flush=True)
                    processed_models.add(model_file)
                    print("="*60, flush=True)
                    continue

                # 将日志文件和输出都指向这个新目录
                eval_log = os.path.join(eval_dir, f'ev_{base_name}.log')
                eval_cmd_list = [
                    sys.executable, "-u", "evaluation.py",
                    "--model_path", model_path,
                    "--generalization", "--gantt",
                    "--run_name", folder_name,
                    "--output_dir", eval_dir,
                ]
                launch_detached_python(eval_cmd_list, eval_log, cwd=main_dir_abs)

                # 启动 debug_marl_behavior.py (保持不变)
                debug_log = os.path.join(debug_dir, f'db_{base_name}.log')
                debug_cmd_list = [
                    sys.executable, "-u", "debug_marl_behavior.py",
                    "--model_path", model_path,
                ]
                launch_detached_python(debug_cmd_list, debug_log, cwd=main_dir_abs)
                
                processed_models.add(model_file)
                print(f"✅ 已为模型 '{model_file}' 触发评估和调试任务。", flush=True)
                print("="*60, flush=True)

        except FileNotFoundError:
            # 模型目录可能尚未被训练脚本创建
            time.sleep(10)
        except Exception as e:
            print(f"🔴 监控过程中发生错误: {e}", flush=True)
            time.sleep(60)

    print("🏁 监控结束（达到超时时间或脚本被中断）。", flush=True)

def launch_background_process(args):
    """
    作为启动器，创建目录和日志路径，并在后台重新启动脚本作为工作进程。
    """
    print(f"✨ 自动化脚本启动器PID: {os.getpid()}", flush=True)

    # 1. 创建主目录
    now = datetime.datetime.now()
    safe_folder_name = args.folder_name.replace(" ", "_").replace("/", "-")
    main_dir_name = now.strftime('%m%d_%H%M') + '_' + safe_folder_name
    os.makedirs(main_dir_name, exist_ok=True)

    # 1.1. 复制关键脚本
    files_to_copy = [
        'environments/w_factory_config.py',
        'environments/w_factory_env.py',
        'mappo/ppo_marl_train.py',
        'mappo/ppo_network.py',
        'mappo/ppo_buffer.py',
        'mappo/ppo_worker.py',
        'mappo/ppo_trainer.py',
        'mappo/sampling_utils.py',
        'debug_marl_behavior.py',
        'evaluation.py',
        'plotting.py',
        'log_parser.py'
    ]
    print(f"📋 正在复制 {len(files_to_copy)} 个关键脚本到 '{main_dir_name}'...", flush=True)
    for file_path in files_to_copy:
        try:
            dst_path = os.path.join(main_dir_name, file_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(file_path, dst_path)
        except Exception as e:
            print(f"   -> 🔴 复制文件 '{file_path}' 时出错: {e}", flush=True)

    # 2. 定义日志文件路径 (使用固定、简洁的名称)
    log_file_name = "auto_train_monitor.log"
    log_file_path = os.path.join(main_dir_name, log_file_name)

    # 3. 构建在后台运行的命令
    # 使用 sys.executable 确保使用相同的Python解释器
    # 使用 -u 标志确保实时输出
    command_str = (
        f"nohup {sys.executable} -u {__file__} "
        f"\"{args.folder_name}\" "
        f"--internal-run "
        f"--main-dir \"{main_dir_name}\" "
        f"--scenario {getattr(args, 'scenario', 'factory')} "
        f"--candidate-source {getattr(args, 'candidate_source', 'endogenous')} "
        f"> \"{log_file_path}\" 2>&1 &"
    )

    print(f"🚀 正在后台启动自动化脚本...")
    proc = subprocess.Popen(command_str, shell=True)
    time.sleep(2)  # 等待片刻以确保进程启动并写入日志
    
    # 尝试从日志文件中提取工作进程的真实 PID
    worker_pid = None
    try:
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                for line in f:
                    if "✨ 自动化工作进程已启动，PID:" in line:
                        worker_pid = line.split("PID:")[-1].strip()
                        break
    except Exception:
        pass
    
    print(f"✅ 自动化流程已在后台开始。您可以关闭此终端。")
    if worker_pid:
        print(f"✨ 自动化工作进程已启动，PID: {worker_pid}")
    print(f"📂 所有输出（包括此脚本的日志）将保存在: {main_dir_name}")
    print(f"📜 使用此命令查看实时日志: tail -f \"{log_file_path}\"")

def run_background_tasks(args):
    """
    作为后台工作进程，执行主要的训练和监控任务。
    """
    # 注册信号处理器，以便在被kill时能够清理子进程
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup) # 处理 Ctrl+C

    main_dir_name = args.main_dir
    folder_name = args.folder_name
    safe_folder_name = folder_name.replace(" ", "_").replace("/", "-")

    main_dir_abs = main_dir_name
    try:
        main_dir_abs = os.path.abspath(main_dir_abs)
    except (OSError, TypeError) as e:
        print(f"Warning: Failed to resolve absolute path for main directory: {e}", flush=True)

    print(f"✨ 自动化工作进程已启动，PID: {os.getpid()}", flush=True)
    print(f"📂 主运行目录: {main_dir_name}", flush=True)

    try:
        os.chdir(main_dir_abs)
    except OSError as e:
        print(f"Warning: Failed to change directory to {main_dir_abs}: {e}", flush=True)

    start_log_parser_watcher(os.path.join(main_dir_abs, "auto_train_monitor.log"), cwd=main_dir_abs)
    
    # 定义模型和日志的输出目录
    models_dir = os.path.join(main_dir_abs, "models")
    logs_dir = os.path.join(main_dir_abs, "logs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 监控 models_dir 以查找由训练脚本创建的新目录
    dirs_before = set(os.listdir(models_dir))

    # 启动训练 (使用包含时间戳和实验名的详细日志)
    now = datetime.datetime.now()
    train_log_name = f"{now.strftime('%m%d_%H%M%S')}_{safe_folder_name}.log"
    train_log = os.path.join(main_dir_abs, train_log_name)
    start_log_parser_watcher(train_log, cwd=main_dir_abs)
    train_cmd_list = [
        sys.executable, "-u", "mappo/ppo_marl_train.py",
        "--models-dir", models_dir,
        "--logs-dir", logs_dir,
        "--scenario", getattr(args, 'scenario', 'factory'),
        "--candidate-source", getattr(args, 'candidate_source', 'endogenous'),
    ]
    launch_and_monitor_child(train_cmd_list, train_log, cwd=main_dir_abs)
    
    time.sleep(10) 

    # 查找由训练脚本创建的新目录
    model_run_dir = find_new_model_dir(models_dir, dirs_before)

    if model_run_dir:
        # 监控目录并启动其他脚本
        monitor_and_launch(model_run_dir, main_dir_abs, folder_name,
                           scenario=getattr(args, 'scenario', 'factory'))
    else:
        print("❌ 未能找到训练输出目录。正在中止监控。", flush=True)
        print(f"   请检查训练日志以获取错误信息: {train_log}", flush=True)

def main():
    """
    主函数，根据参数决定是作为启动器还是作为后台工作进程。
    """
    parser = argparse.ArgumentParser(
        description="自动化MARL模型的训练、评估和调试流程。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "folder_name",
        type=str,
        help="为本次训练运行提供一个描述性名称 (例如, '更改奖励函数测试')。"
    )
    # 添加内部参数，用户无需关心
    parser.add_argument(
        "--internal-run", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--main-dir", type=str, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--scenario", type=str, default="factory", choices=["factory", "delivery"],
        help="训练场景：factory=工厂生产调度；delivery=运力商圈配送调度（比赛）"
    )
    parser.add_argument(
        "--candidate-source", type=str, default="endogenous", choices=["endogenous", "upstream"],
        help="配送场景候选来源：endogenous=环境自采样；upstream=上游（双塔+W&D）精排候选"
    )
    args = parser.parse_args()

    if args.internal_run:
        # 如果有内部运行标记，则执行后台任务
        run_background_tasks(args)
    else:
        # 否则，作为启动器，在后台重新启动自己
        launch_background_process(args)

if __name__ == "__main__":
    # 确保脚本从项目根目录运行
    if not os.path.exists('mappo/ppo_marl_train.py'):
        print("❌ 错误: 此脚本必须从 'MARL_FOR_W_Factory' 项目根目录运行。", flush=True)
        sys.exit(1)
    main()
