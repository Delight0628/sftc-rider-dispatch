"""
MAPPO经验缓冲区模块
==================
实现经验数据的存储和GAE优势函数计算
"""

import numpy as np
from typing import Optional


class ExperienceBuffer:
    """
    MAPPO经验缓冲区
    ----------------
    存储多智能体交互经验，支持GAE优势函数计算
    
    特性：
    - 支持局部观测和全局状态
    - 处理episode截断和终止
    - 智能优势函数标准化
    """
    
    def __init__(self):
        self.states = []
        self.global_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.action_probs = []
        self.dones = []
        self.truncated = []

    def store(self, state, global_state, action, reward, value, action_prob, done, truncated=False):
        """存储单步经验数据"""
        self.states.append(state)
        self.global_states.append(global_state)
        self.actions.append(np.array(action))  # 统一转换为numpy数组
        self.rewards.append(reward)
        self.values.append(value)
        self.action_probs.append(action_prob)
        self.dones.append(done)
        self.truncated.append(truncated)

    def get_batch(self, gamma=0.99, lam=0.95, next_value_if_truncated=None, advantage_clip_val: Optional[float] = None):
        """
        计算GAE优势函数并返回训练批次
        
        使用Generalized Advantage Estimation (GAE)算法计算优势函数，
        正确处理episode截断和终止，避免价值泄漏。
        
        Args:
            gamma: 折扣因子 (默认0.99)
            lam: GAE平滑参数 (默认0.95)
            next_value_if_truncated: 截断时的bootstrap价值估计
            advantage_clip_val: 优势函数裁剪阈值
            
        Returns:
            tuple: (states, global_states, actions, action_probs, advantages, returns)
        """
        # 转换为numpy数组
        states = np.array(self.states, dtype=np.float32)
        global_states = np.array(self.global_states, dtype=np.float32)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        action_probs = np.array(self.action_probs, dtype=np.float32)
        dones = np.array(self.dones)
        truncateds = np.array(self.truncated)
        
        # GAE计算：从后向前遍历
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            # 确定下一状态的价值估计（bootstrap 与 递推掩码分离）
            if t == len(rewards) - 1:
                # 轨迹末端
                if dones[t]:
                    next_value = 0.0  # 正常终止：价值为0
                elif truncateds[t]:
                    # 截断：若提供外部bootstrap值则使用，否则回退为0
                    next_value = next_value_if_truncated if next_value_if_truncated is not None else 0.0
                else:
                    # 非终止且未标记截断：无法观测下一价值，回退到可选bootstrap或0
                    next_value = next_value_if_truncated if next_value_if_truncated is not None else 0.0
            else:
                # 轨迹中间：使用下一时刻的价值估计
                next_value = values[t + 1]

            # 分离两类掩码：
            # - bootstrap_mask：允许在截断步使用价值bootstrap，但在终止步关闭
            # - recursion_mask：终止或截断都阻断优势的递推项
            bootstrap_mask = (1 - dones[t])
            recursion_mask = (1 - dones[t]) * (1 - truncateds[t])

            # GAE递推：δ_t + γλ * A_{t+1}
            delta = rewards[t] + gamma * next_value * bootstrap_mask - values[t]
            advantages[t] = delta + gamma * lam * recursion_mask * last_advantage
            last_advantage = advantages[t]
        
        # 计算目标回报
        returns = advantages + values
        
        # 优势函数标准化（提升训练稳定性）
        if len(advantages) > 1:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages)
            if adv_std > 1e-6:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = advantages - adv_mean
        
        # 可选的优势裁剪（防止极端值）
        if advantage_clip_val is not None:
            advantages = np.clip(advantages, -advantage_clip_val, advantage_clip_val)
        
        return states, global_states, actions, action_probs, advantages, returns
    
    def clear(self):
        self.states.clear()
        self.global_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.action_probs.clear()
        self.dones.clear()
        self.truncated.clear()

    def __len__(self):
        return len(self.states)

