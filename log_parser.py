import re
import pandas as pd
import argparse
from typing import List, Dict, Any
import os

def parse_log_file(log_path: str) -> List[Dict[str, Any]]:
    """
    解析PPO训练日志文件，提取每个回合的关键指标。

    Args:
        log_path: 日志文件的路径。

    Returns:
        一个包含每回合数据的字典列表。
    """
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 兼容当前 ppo_trainer 的输出格式（见 mappo/ppo_trainer.py 的 line1/line2/line3）
    re_episode = re.compile(
        r"(?:🔂\s*)?(?:训练回合|回合)\s*(\d+)\s*/\s*(\d+).*?平均奖励:\s*([-\d.]+).*?Actor损失:\s*([-\d.]+).*?(?:本轮用[时時]|本轮用时):\s*([\d.]+)s"
    )
    re_kpi = re.compile(
        r"(?:📊\s*)?(?:此回合KPI评估|KPI).*?总完工时间:\s*([\d.]+)min.*?(?:设备利用率|利用率):\s*([\d.]+)%.*?(?:订单延期时间|延期时间):\s*([\d.]+)min.*?完成零件数:\s*([\d.]+)\s*/\s*(\d+)"
    )
    re_score = re.compile(
        r"(?:🚥\s*)?回合评分:\s*([\d.]+).*?\(全局最佳:\s*([\d.]+)\)"
    )
    re_stage_course = re.compile(r"课程:\s*'([^']+)'", re.UNICODE)
    re_stage_simple = re.compile(r"阶段:\s*'([^']+)'", re.UNICODE)
    re_eval_env = re.compile(r"评估环境:\s*\[([^\]]+)\]", re.UNICODE)

    episodes_data: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}

    def _flush_current():
        nonlocal cur
        if cur and ('回合' in cur):
            episodes_data.append(cur)
        cur = {}

    for ln in lines:
        line = ln.strip()
        if not line:
            continue

        m = re_episode.search(line)
        if m:
            _flush_current()
            ep, ep_total, reward, actor_loss, it_time = m.groups()
            cur = {
                '课程阶段': '未知',
                '回合': int(ep),
                '奖励': float(reward),
                'Actor损失': float(actor_loss),
                '本轮用时_s': float(it_time),
                '总完工时间_min': None,
                '设备利用率_%': None,
                '订单延期时间_min': None,
                '完成零件数': None,
                '目标零件数': None,
                '回合评分': None,
                '全局最佳评分': None,
                '阶段最佳评分': None,
                '模型是否更新': 0,
                '评估环境': None,
            }
            continue

        if cur:
            mk = re_kpi.search(line)
            if mk:
                makespan, util_pct, tard, comp, target = mk.groups()
                cur['总完工时间_min'] = float(makespan)
                cur['设备利用率_%'] = float(util_pct)
                cur['订单延期时间_min'] = float(tard)
                try:
                    cur['完成零件数'] = int(float(comp))
                except Exception:
                    cur['完成零件数'] = None
                cur['目标零件数'] = int(target)

                me = re_eval_env.search(line)
                if me:
                    cur['评估环境'] = me.group(1).strip()

                st = None
                mc = re_stage_course.search(line)
                if mc:
                    st = mc.group(1)
                else:
                    ms = re_stage_simple.search(line)
                    if ms:
                        st = ms.group(1)
                if st:
                    cur['课程阶段'] = st
                continue

            ms = re_score.search(line)
            if ms:
                score, best_global = ms.groups()
                cur['回合评分'] = float(score)
                cur['全局最佳评分'] = float(best_global)
                # “阶段最佳”在不同模式下字段名不同，这里做一个宽松提取
                m_stage_best = re.search(r"\((?:基础阶段最佳|泛化阶段最佳|阶段最佳):\s*([\d.]+)\)", line)
                if m_stage_best:
                    cur['阶段最佳评分'] = float(m_stage_best.group(1))
                cur['模型是否更新'] = 1 if ('✅' in line) else 0
                continue

    _flush_current()

    # 过滤掉明显不完整的行（例如没有任何评分且没有KPI）
    cleaned = []
    for d in episodes_data:
        if d.get('总完工时间_min') is None and d.get('回合评分') is None:
            continue
        cleaned.append(d)
    return cleaned

def main():
    """主函数，用于解析命令行参数并执行日志解析。"""
    parser = argparse.ArgumentParser(
        description="将PPO训练日志文件解析并转换为同名的Excel文件。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="输入的日志文件路径, 例如: 'GPUtraining_log_20250827_232144.log'"
    )
    args = parser.parse_args()

    # 从日志文件名自动生成输出的Excel文件名
    base_name = os.path.splitext(args.log_file)[0]
    output_excel = f"{base_name}.xlsx"

    print(f"正在解析日志文件: {args.log_file}...")
    try:
        data = parse_log_file(args.log_file)
        if not data:
            print("❌ 错误：在日志文件中没有找到任何有效的回合数据。请检查文件内容和格式。")
            return
            
        df = pd.DataFrame(data)
        
        print(f"成功解析 {len(df)} 条回合数据。")
        print(f"正在将数据写入Excel文件: {output_excel}...")
        
        df.to_excel(output_excel, index=False, engine='openpyxl')
        
        print(f"✅ 成功！数据已保存到: {output_excel}")
        
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{args.log_file}'。请检查路径是否正确。")
    except Exception as e:
        print(f"❌ 解析过程中发生未知错误: {e}")


if __name__ == '__main__':
    main()
