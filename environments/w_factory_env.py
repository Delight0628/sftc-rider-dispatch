"""
W工厂生产调度系统 - 仿真环境核心
包含SimPy仿真逻辑和PettingZoo多智能体环境接口
"""

import simpy
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers
from .w_factory_config import *


SILENT_MODE = True  # 设置为True时，完全禁用调试输出

# --- V3 融合版：新增的辅助函数 ---
def _calculate_part_total_remaining_processing_time(part: 'Part') -> float:
    """计算一个零件所有剩余工序的总加工时间"""
    route = get_route_for_product(part.product_type)
    if part.current_step >= len(route):
        return 0.0
    return sum(step['time'] for i, step in enumerate(route) if i >= part.current_step)

def calculate_slack_time(part: 'Part', current_time: float, queues: Dict[str, Any] = None, workstations: Dict[str, Dict] = None) -> float:
    """
    计算零件的松弛时间 (Slack Time) - 改进版本
    
    Args:
        part: 零件对象
        current_time: 当前时间
        queues: 工作站队列字典（可选）
        workstations: 工作站配置字典（可选）
    
    Returns:
        松弛时间（分钟）。正值表示有余量，负值表示可能延期
    """
    remaining_processing_time = _calculate_part_total_remaining_processing_time(part)
    
    # 基础松弛时间（原始计算）
    basic_slack = (part.due_date - current_time) - remaining_processing_time
    
    return basic_slack

# =============================================================================
# 1. 数据结构定义 (Data Structures)
# =============================================================================

class Part:
    """零件类 - 表示生产中的一个零件"""
    def __init__(self, part_id: int, product_type: str, order_id: int, 
                 due_date: float, priority: int):
        self.part_id = part_id
        self.product_type = product_type
        self.order_id = order_id
        self.due_date = due_date
        self.priority = priority
        self.current_step = 0
        self.start_time = 0
        self.completion_time = None
        self.contribution_map: Dict[str, float] = {}
        
    def get_current_station(self) -> Optional[str]:
        """获取当前需要加工的工作站"""
        route = get_route_for_product(self.product_type)
        if self.current_step < len(route):
            return route[self.current_step]["station"]
        return None
    
    def get_processing_time(self) -> float:
        """获取当前工序的加工时间"""
        route = get_route_for_product(self.product_type)
        if self.current_step < len(route):
            return route[self.current_step]["time"]
        return 0
    
    def is_completed(self) -> bool:
        """检查零件是否完成所有工序"""
        route = get_route_for_product(self.product_type)
        return self.current_step >= len(route)

class Order:
    """订单类"""
    def __init__(self, order_id: int, product: str, quantity: int, 
                 priority: int, due_date: float, arrival_time: float = 0):
        self.order_id = order_id
        self.product = product
        self.quantity = quantity
        self.priority = priority
        self.due_date = due_date
        self.arrival_time = arrival_time
        self.parts = []
        self.completed_parts = 0
        
    def create_parts(self) -> List[Part]:
        """为订单创建零件"""
        self.parts = []
        for i in range(self.quantity):
            part_id = self.order_id * 1000 + i
            part = Part(part_id, self.product, self.order_id, 
                       self.due_date, self.priority)
            part.start_time = self.arrival_time
            self.parts.append(part)
        return self.parts

# =============================================================================
# 2. SimPy仿真核心 (SimPy Simulation Core)
# =============================================================================

class WFactorySim:
    """W工厂仿真核心类 - 基于SimPy的离散事件仿真"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 允许从config覆盖仿真时间尺度/超时上限（默认保持与w_factory_config一致）
        try:
            self._simulation_time = float(self.config.get('SIMULATION_TIME', SIMULATION_TIME))
        except Exception:
            self._simulation_time = float(SIMULATION_TIME)

        try:
            self._simulation_timeout_multiplier = float(
                self.config.get('SIMULATION_TIMEOUT_MULTIPLIER', SIMULATION_TIMEOUT_MULTIPLIER)
            )
        except Exception:
            self._simulation_timeout_multiplier = float(SIMULATION_TIMEOUT_MULTIPLIER)

        try:
            default_max_time = float(self._simulation_time) * float(self._simulation_timeout_multiplier)
            self._max_sim_time = float(self.config.get('MAX_SIM_TIME', default_max_time))
        except Exception:
            self._max_sim_time = float(SIMULATION_TIME) * float(SIMULATION_TIMEOUT_MULTIPLIER)

        try:
            _max_machine_count = max(int(v.get('count', 1)) for v in WORKSTATIONS.values()) if WORKSTATIONS else 1
        except Exception:
            _max_machine_count = 1
        self._capacity_norm_denom = float(max(1, _max_machine_count))

        try:
            _default_queue_len_norm = max(10.0, float(QUEUE_CAPACITY) / float(max(1, len(WORKSTATIONS))))
            self._queue_len_norm = float(self.config.get('w_station_capacity_norm', _default_queue_len_norm))
        except Exception:
            self._queue_len_norm = float(ENHANCED_OBS_CONFIG.get("w_station_capacity_norm", 10.0))

        # 定义智能体列表
        self.agents = [f"agent_{station}" for station in WORKSTATIONS.keys()]
        
        # 调试级别控制
        self.debug_level = self.config.get('debug_level', 'INFO')  # DEBUG, INFO, WARNING, ERROR
        
        # 训练模式标志，控制输出冗余度
        self._training_mode = self.config.get('training_mode', False)
        
        # 减少输出冗余
        if self._training_mode:
            self.debug_level = 'WARNING'
        
        # 10-27-16-30 修复：统一兼容 'disable_failures' 配置键（应用/评估端常用），并读取动态事件开关
        # 允许在不同训练阶段启用/禁用设备故障和紧急插单
        self._equipment_failure_enabled = bool(self.config.get('equipment_failure_enabled', False))
        # 10-27-16-30 若传入 'disable_failures'=True，则强制关闭设备故障
        if 'disable_failures' in self.config:
            try:
                self._equipment_failure_enabled = not bool(self.config.get('disable_failures'))
            except Exception:
                self._equipment_failure_enabled = False
        self._emergency_orders_enabled = bool(self.config.get('emergency_orders_enabled', False))
        
        # 12-02 新增：读取设备故障和紧急插单的高级配置参数
        self._equipment_failure_config = self.config.get('equipment_failure_config', {})
        self._emergency_orders_config = self.config.get('emergency_orders_config', {})
        
        # 仿真环境
        self.env = simpy.Environment()
        self.current_time = 0
        self.simulation_ended = False
        
        # 状态跟踪
        self.active_parts: List[Part] = []
        self.completed_parts: List[Part] = []
        self.orders: List[Order] = []
        
        # 资源和队列
        self.queues: Dict[str, simpy.Store] = {}
        self.resources: Dict[str, simpy.Resource] = {}
        self.equipment_status: Dict[str, Dict[str, Any]] = {}
        
        # 新增：用于生成甘特图的加工历史记录
        self.gantt_chart_history: List[Dict[str, Any]] = []

        # 新增：动态事件时间线（用于UI标注与回放）
        # 记录格式（可JSON序列化）：
        # - 故障：{"type":"failure","station":str,"start":float,"end":float}
        # - 插单：{"type":"emergency_order","time":float,"order_id":int,"product":str,"quantity":int,"priority":int,"due_date":float}
        self.event_timeline: List[Dict[str, Any]] = []
        
        # 性能指标
        self._start_times: Dict[int, float] = {}
        self._end_times: Dict[int, float] = {}
        self._equipment_busy_time: Dict[str, float] = defaultdict(float)
        
        # 订单级别跟踪系统
        self.order_progress = {}  # 订单进度跟踪
        self.order_completion_times = {}  # 订单完成时间
        
        self.stats: Dict[str, Any] = {
            'last_completed_count': 0,
            'completed_orders': 0,
            'last_completed_orders': 0,
            'makespan': 0,
            'total_tardiness': 0,
            'max_tardiness': 0,
            'equipment_utilization': {},
            'queue_lengths': defaultdict(list),
            'total_parts': 0,
            'idle_when_work_available_count': 0,
            'equipment_failure_event_count': 0,
            'equipment_failure_event_count_by_station': {},
            'emergency_orders_inserted_count': 0,
            'emergency_parts_inserted_count': 0
        }

        # 终局奖励发放标记（防重复）
        self.final_bonus_awarded = False
        self.final_bonus_value = 0.0

        # 🔧 新增：迟期总量缓存与候选缓存（保证同一步一致性）
        self._last_overdue_sum: float = 0.0
        self._cached_candidates: Dict[str, List[Dict[str, Any]]] = {}
        self._initial_target_parts: int = 0
        self._last_score_potential: float = 0.0
        
        # 🔧 新增：进度和紧急度追踪（用于新奖励系统）
        self._last_progress_ratio: float = 0.0
        self._last_urgency_sum: float = 0.0
        # 🔧 修改：候选动作动态范围
        self._candidate_action_start: int = 1  # 从动作1开始（动作0是IDLE）
        self._candidate_action_end: int = int(ENHANCED_OBS_CONFIG.get("num_candidate_workpieces", 0))
        
        # 用于快速查找下游工作站的缓存
        self._downstream_map = self._create_downstream_map()
        
        self._initialize_resources()
        
        # --- 方案三：引入环境随机性 ---
        # 备份基础订单，以便在重置时重新引入随机性
        self._base_orders_template = [o.copy() for o in BASE_ORDERS]
        self._initialize_orders()

        self._init_score_decomposition_tracking()

        # 10-27-16-30 新增：若启用紧急插单，则启动插单生成进程
        if self._emergency_orders_enabled:
            self.env.process(self._emergency_order_process())

        # 🔧 新增：候选采样策略（评估可设为确定性，保证启发式复现性）
        self._deterministic_candidates = bool(self.config.get('deterministic_candidates', False))
    
    def reset(self):
        """重置仿真状态"""
        self.env = simpy.Environment()
        self.current_time = 0
        self.simulation_ended = False
        
        # 清空所有状态
        self.active_parts.clear()
        self.completed_parts.clear()
        self.orders.clear()
        self.queues.clear()
        self.resources.clear()
        self.equipment_status.clear()
        
        # 新增：清空甘特图历史
        self.gantt_chart_history.clear()

        # 新增：清空事件时间线
        self.event_timeline.clear()
        
        # 重置订单跟踪
        self.order_progress.clear()
        self.order_completion_times.clear()
        
        # 重新初始化
        self._initialize_resources()
        self._initialize_orders()

        self._init_score_decomposition_tracking()

        # 10-27-16-30 新增：reset 后重新启动紧急插单进程
        if bool(self.config.get('emergency_orders_enabled', False)):
            self.env.process(self._emergency_order_process())
        
        # 完整重置stats字典
        self.stats = {
            'last_completed_count': 0,
            'completed_orders': 0,
            'last_completed_orders': 0,
            'makespan': 0,
            'total_tardiness': 0,
            'max_tardiness': 0,
            'equipment_utilization': {},
            'queue_lengths': defaultdict(list),
            'total_parts': 0,
            'idle_when_work_available_count': 0,
            'equipment_failure_event_count': 0,
            'equipment_failure_event_count_by_station': {},
            'emergency_orders_inserted_count': 0,
            'emergency_parts_inserted_count': 0
        }

        # 重置终局奖励标记
        self.final_bonus_awarded = False
        self.final_bonus_value = 0.0

        # 重置迟期与候选缓存
        self._last_overdue_sum = 0.0
        self._cached_candidates.clear()
        
        # 重置进度和紧急度追踪
        self._last_progress_ratio = 0.0
        self._last_urgency_sum = 0.0
    
    def _init_score_decomposition_tracking(self):
        try:
            self._initial_target_parts = int(sum(int(o.quantity) for o in self.orders)) if self.orders else 0
        except Exception:
            self._initial_target_parts = 0

        if REWARD_CONFIG.get('score_decomposition_shaping_enabled', False):
            try:
                weights = REWARD_CONFIG.get('score_decomposition_shaping_weights', {}) or {}
                self._last_score_potential = float(self._compute_score_decomposition_potential(weights))
            except Exception:
                self._last_score_potential = 0.0
        else:
            self._last_score_potential = 0.0

    def _estimate_total_tardiness_now(self, current_time: float) -> float:
        total_tardiness = 0.0
        for order in self.orders:
            completion_time = self.order_completion_times.get(order.order_id)
            if completion_time is not None:
                if completion_time > order.due_date:
                    total_tardiness += float(completion_time - order.due_date)
            else:
                if current_time > order.due_date:
                    total_tardiness += float(current_time - order.due_date)
        return float(total_tardiness)

    def _estimate_mean_utilization_now(self, current_time: float) -> float:
        if current_time <= 0:
            return 0.0

        util_values = []
        for station_name in WORKSTATIONS.keys():
            status = self.equipment_status.get(station_name, {})
            capacity = WORKSTATIONS[station_name]['count']
            if capacity <= 0:
                util_values.append(0.0)
                continue

            busy_machine_time = float(status.get('busy_machine_time', 0.0))
            last_event_time = float(status.get('last_event_time', 0.0))
            busy_count = float(status.get('busy_count', 0.0))
            if current_time > last_event_time:
                busy_machine_time += (current_time - last_event_time) * busy_count

            util_values.append(float(busy_machine_time / (float(current_time) * float(capacity))))

        return float(np.mean(util_values)) if util_values else 0.0

    def _compute_score_decomposition_potential(self, weights: Dict[str, float]) -> float:
        current_time = float(self.env.now)
        _t = float(self._simulation_time) if float(self._simulation_time) > 0 else float(SIMULATION_TIME)

        target_parts = int(self._initial_target_parts) if int(self._initial_target_parts) > 0 else 0
        completed_parts = int(len(self.completed_parts))
        completion_score = (float(completed_parts) / float(target_parts)) if target_parts > 0 else 0.0
        if completion_score > 1.0:
            completion_score = 1.0
        if completion_score < 0.0:
            completion_score = 0.0

        tardiness = float(self._estimate_total_tardiness_now(current_time))
        tardiness_score = max(0.0, 1.0 - tardiness / float(_t * 2.0))
        makespan_score = max(0.0, 1.0 - current_time / float(_t * 1.5))
        utilization_score = float(self._estimate_mean_utilization_now(current_time))

        w_completion = float(weights.get('completion', 0.40))
        w_tardiness = float(weights.get('tardiness', 0.35))
        w_makespan = float(weights.get('makespan', 0.15))
        w_util = float(weights.get('utilization', 0.10))

        return (
            w_completion * float(completion_score) +
            w_tardiness * float(tardiness_score) +
            w_makespan * float(makespan_score) +
            w_util * float(utilization_score)
        )
    
    def _initialize_resources(self):
        """初始化设备资源和队列"""
        for station_name, station_config in WORKSTATIONS.items():
            # 创建SimPy资源（设备）
            capacity = station_config["count"]
            self.resources[station_name] = simpy.Resource(self.env, capacity=capacity)
            
            # 创建输入队列
            self.queues[station_name] = simpy.Store(self.env, capacity=QUEUE_CAPACITY)
            
            # 初始化设备状态
            self.equipment_status[station_name] = {
                'busy_count': 0,
                'total_busy_time': 0,
                'last_status_change': 0,
                'is_failed': False,
                'failure_end_time': 0,
                # 新增：用于精确统计并行设备的忙碌"面积"（机器-分钟）
                'busy_machine_time': 0.0,
                'last_event_time': 0.0,
            }
            
            # 启动设备处理进程
            self.env.process(self._equipment_process(station_name))
    
    def _initialize_orders(self):
        """初始化订单（支持课程学习、自定义订单和环境随机性）"""
        # 🔧 修复：优先使用自定义订单配置
        if 'custom_orders' in self.config:
            # 使用自定义订单，忽略课程学习缩放
            actual_orders_config = self.config['custom_orders']
            # 修复：即使使用custom_orders，也应尊重randomize_env开关
            is_randomized = bool(self.config.get('randomize_env', False))
        else:
            # --- 方案三：引入环境随机性 ---
            orders_scale = self.config.get('orders_scale', 1.0)
            time_scale = self.config.get('time_scale', 1.0)
            is_randomized = self.config.get('randomize_env', False)

            base_orders_template = self._base_orders_template
            
            # 如果启用课程学习，按比例调整订单
            actual_orders_config = []
            if orders_scale < 1.0:
                # 计算需要多少个零件
                total_parts_needed = int(sum(o["quantity"] for o in BASE_ORDERS) * orders_scale)
                parts_added = 0
                
                # 优先选择不同产品类型的订单，保持多样性
                for order_data in BASE_ORDERS:
                    if parts_added >= total_parts_needed:
                        break
                    
                    # 调整订单数量
                    adjusted_quantity = min(order_data["quantity"], total_parts_needed - parts_added)
                    if adjusted_quantity > 0:
                        adjusted_order = order_data.copy()
                        adjusted_order["quantity"] = adjusted_quantity
                        adjusted_order["due_date"] = order_data["due_date"] * time_scale  # 放宽时间限制
                        actual_orders_config.append(adjusted_order)
                        parts_added += adjusted_quantity
            else:
                actual_orders_config = base_orders_template

        # 创建订单对象
        for i, order_data in enumerate(actual_orders_config):
            order_data_copy = order_data.copy()

            # --- 方案三：如果启用了随机化，则添加扰动 ---
            if is_randomized:
                due_date_jitter_range = ENV_RANDOMIZATION_CONFIG.get("due_date_jitter", 15.0)
                arrival_time_jitter_range = ENV_RANDOMIZATION_CONFIG.get("arrival_time_jitter", 10.0)
                
                due_date_jitter = np.random.uniform(-due_date_jitter_range, due_date_jitter_range)
                arrival_time_jitter = np.random.uniform(0, arrival_time_jitter_range)
                
                order_data_copy['due_date'] += due_date_jitter
                # 修复：使用订单索引作为基础到达时间，而不是不存在的'start_time'
                base_arrival_time = order_data_copy.get('arrival_time', 0)
                order_data_copy['arrival_time'] = base_arrival_time + arrival_time_jitter
            else:
                # 修复：确保有默认的到达时间
                order_data_copy['arrival_time'] = order_data_copy.get('arrival_time', 0)

            order = Order(
                order_id=i,
                product=order_data_copy["product"],
                quantity=order_data_copy["quantity"],
                priority=order_data_copy["priority"],
                due_date=order_data_copy["due_date"],
                arrival_time=order_data_copy['arrival_time']
            )
            self.orders.append(order)
            
            # 创建零件并添加到仿真中
            parts = order.create_parts()
            for part in parts:
                self.env.process(self._part_process(part))
                self.active_parts.append(part)
    
    def _part_process(self, part: Part):
        """零件的生产流程进程 - 简化版本"""
        # 在达到计划到达时间前等待
        if hasattr(part, 'start_time') and part.start_time > self.env.now:
            yield self.env.timeout(part.start_time - self.env.now)
        # 将零件放入第一个工作站的队列
        first_station = part.get_current_station()
        if first_station:
            yield self.queues[first_station].put(part)

    def _equipment_process(self, station_name: str):
        """设备处理进程 - 处理设备故障等事件"""
        while True:
            # 10-23-18-00 修改：使用实例级别的配置而非全局配置
            # 这允许不同worker在同一进程中使用不同的故障配置
            if self._equipment_failure_enabled:
                # 12-02 新增：支持自定义设备故障参数
                mtbf_hours = self._equipment_failure_config.get('mtbf_hours', EQUIPMENT_FAILURE["mtbf_hours"])
                mttr_minutes = self._equipment_failure_config.get('mttr_minutes', EQUIPMENT_FAILURE["mttr_minutes"])
                failure_prob = self._equipment_failure_config.get('failure_probability', EQUIPMENT_FAILURE["failure_probability"])
                
                # 随机设备故障
                failure_interval = np.random.exponential(mtbf_hours * 60)
                yield self.env.timeout(failure_interval)
                
                if random.random() < failure_prob:
                    # 设备故障
                    try:
                        self.stats['equipment_failure_event_count'] = int(self.stats.get('equipment_failure_event_count', 0)) + 1
                        by_station = self.stats.get('equipment_failure_event_count_by_station', {})
                        if not isinstance(by_station, dict):
                            by_station = {}
                        by_station[station_name] = int(by_station.get(station_name, 0)) + 1
                        self.stats['equipment_failure_event_count_by_station'] = by_station
                    except Exception:
                        pass
                    self.equipment_status[station_name]['is_failed'] = True
                    repair_time = np.random.exponential(mttr_minutes)
                    self.equipment_status[station_name]['failure_end_time'] = (
                        self.env.now + repair_time
                    )

                    # 记录故障事件区间（用于甘特图标注）
                    try:
                        self.event_timeline.append({
                            'type': 'failure',
                            'station': str(station_name),
                            'start': float(self.env.now),
                            'end': float(self.env.now + repair_time),
                        })
                    except Exception:
                        pass
                    
                    yield self.env.timeout(repair_time)
                    self.equipment_status[station_name]['is_failed'] = False
            else:
                # 静态训练模式：设备不会故障，只需要等待仿真结束
                yield self.env.timeout(float(self._simulation_time))

    # 10-27-16-30 新增：紧急插单生成进程
    def _emergency_order_process(self):
        """根据配置按泊松过程向系统注入紧急订单。"""
        while True:
            if not self._emergency_orders_enabled:
                # 未启用时，避免忙等
                yield self.env.timeout(float(self._simulation_time))
                continue

            # 12-02 新增：支持自定义紧急插单参数
            arrival_rate_per_hour = self._emergency_orders_config.get('arrival_rate', EMERGENCY_ORDERS.get('arrival_rate', 0.0))
            if arrival_rate_per_hour <= 0.0:
                # 无到达，直接等待至仿真结束
                yield self.env.timeout(float(self._simulation_time))
                continue
            inter_arrival = np.random.exponential(60.0 / arrival_rate_per_hour)
            yield self.env.timeout(inter_arrival)

            # 10-27-16-30 生成紧急订单参数
            try:
                product = random.choice(list(PRODUCT_ROUTES.keys()))
                # 小批量插单，避免过度干扰基础流
                base_qty = 0
                for order_data in BASE_ORDERS:
                    try:
                        if order_data.get("product") == product:
                            q = int(order_data.get("quantity", 0))
                            if q > base_qty:
                                base_qty = q
                    except Exception:
                        continue
                if base_qty <= 0:
                    base_qty = 3
                max_fraction = 0.3
                max_emerg_qty = max(1, int(np.ceil(base_qty * max_fraction)))
                quantity = int(np.random.randint(1, max_emerg_qty + 1))
                base_priority = 2
                priority_boost = int(self._emergency_orders_config.get('priority_boost', EMERGENCY_ORDERS.get('priority_boost', 0)))
                priority = int(np.clip(base_priority + priority_boost, 1, 5))

                # 交期：基于总加工时间的缩短比例
                route = get_route_for_product(product)
                per_item_time = sum(step['time'] for step in route)
                due_reduction = float(self._emergency_orders_config.get('due_date_reduction', EMERGENCY_ORDERS.get('due_date_reduction', 0.7)))
                # 至少留一段缓冲（30分钟）
                due_date = self.env.now + max(30.0, per_item_time * quantity * due_reduction)

                # 分配新订单ID
                next_order_id = (max([o.order_id for o in self.orders]) + 1) if self.orders else 0
                emerg_order = Order(
                    order_id=next_order_id,
                    product=product,
                    quantity=quantity,
                    priority=priority,
                    due_date=due_date,
                    arrival_time=self.env.now
                )
                self.orders.append(emerg_order)

                try:
                    self.stats['emergency_orders_inserted_count'] = int(self.stats.get('emergency_orders_inserted_count', 0)) + 1
                    self.stats['emergency_parts_inserted_count'] = int(self.stats.get('emergency_parts_inserted_count', 0)) + int(quantity)
                except Exception:
                    pass

                # 创建零件并注入首工位队列
                for part in emerg_order.create_parts():
                    part.start_time = self.env.now  # 立即到达
                    self.env.process(self._part_process(part))
                    self.active_parts.append(part)

                # 记录插单事件时间点（用于甘特图标注）
                try:
                    self.event_timeline.append({
                        'type': 'emergency_order',
                        'time': float(self.env.now),
                        'order_id': int(emerg_order.order_id),
                        'product': str(product),
                        'quantity': int(quantity),
                        'priority': int(priority),
                        'due_date': float(due_date),
                    })
                except Exception:
                    pass
            except Exception:
                # 插单失败不应中断主仿真
                pass
    
    def _update_equipment_status(self, station_name: str, busy: bool):
        """更新设备状态"""
        status = self.equipment_status[station_name]
        current_time = self.env.now
        
        # 在变更 busy_count 之前，先结算从上次事件到现在的忙碌"面积"
        previous_busy_count = status['busy_count']
        last_event_time = status.get('last_event_time', 0.0)
        if current_time > last_event_time:
            elapsed = current_time - last_event_time
            # 积分：elapsed * previous_busy_count（机器-分钟）
            status['busy_machine_time'] = status.get('busy_machine_time', 0.0) + elapsed * previous_busy_count
        status['last_event_time'] = current_time
        
        if busy:
            status['busy_count'] += 1
        else:
            status['busy_count'] = max(0, status['busy_count'] - 1)
            # 累计忙碌时间
            if status['busy_count'] == 0:
                status['total_busy_time'] += (
                    current_time - status['last_status_change']
                )
        
        status['last_status_change'] = current_time
    
    def _create_downstream_map(self) -> Dict[str, str]:
        """创建下游工作站映射，用于快速查询"""
        downstream_map = {}
        routes = list(PRODUCT_ROUTES.values())
        for route in routes:
            for i in range(len(route) - 1):
                current_station = route[i]["station"]
                next_station = route[i+1]["station"]
                if current_station not in downstream_map:
                    downstream_map[current_station] = next_station
        return downstream_map
    
    def _update_order_progress(self):
        """更新订单进度跟踪"""
        for order in self.orders:
            completed_parts = sum(1 for part in self.completed_parts 
                                if part.order_id == order.order_id)
            progress_rate = completed_parts / order.quantity if order.quantity > 0 else 0
            self.order_progress[order.order_id] = progress_rate
            
            # 检查订单是否完成
            if progress_rate >= 1.0 and order.order_id not in self.order_completion_times:
                self.order_completion_times[order.order_id] = self.current_time
                self.stats['completed_orders'] += 1

    def _update_completion_stats(self, part: Part):
        """更新完成统计"""
        # 计算延期
        tardiness = max(0, part.completion_time - part.due_date)
        self.stats['total_tardiness'] += tardiness
        self.stats['max_tardiness'] = max(self.stats['max_tardiness'], tardiness)
        
        # 更新makespan
        self.stats['makespan'] = max(self.stats['makespan'], part.completion_time)
        
        self.stats['total_parts'] += 1
    
    def get_state_for_agent(self, agent_id: str) -> np.ndarray:
        """
        方案B：全局优化观测状态
        - 包含四大部分：
          1. 智能体自身特征 (8维)
          2. 全局宏观特征 (4维)
          3. 当前队列摘要统计 (30维)
          4. 候选工件详细特征 (90维)
        """
        station_name = agent_id.replace("agent_", "")
        
        # --- 1. 智能体自身特征 (Agent Features) - 8维 ---
        agent_features_list = []
        station_types = list(WORKSTATIONS.keys())
        station_index = station_types.index(station_name)
        agent_features_list.extend([1.0 if i == station_index else 0.0 for i in range(len(station_types))])
        
        capacity = WORKSTATIONS[station_name]['count']
        agent_features_list.append(float(capacity) / float(self._capacity_norm_denom))  # 归一化能力
        
        busy_ratio = self.equipment_status[station_name]['busy_count'] / capacity
        agent_features_list.append(busy_ratio)
        agent_features_list.append(1.0 if self.equipment_status[station_name]['is_failed'] else 0.0)
        
        agent_features = np.array(agent_features_list, dtype=np.float32)

        # 保留：时间进度、WIP率、瓶颈拥堵度、当前队列长度（4维中性信息）
        _t = float(self._simulation_time) if float(self._simulation_time) > 0 else float(SIMULATION_TIME)
        time_normalized = self.env.now / _t
        total_parts_in_system = sum(order.quantity for order in self.orders)
        wip_normalized = len(self.active_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        
        # 瓶颈工作站拥堵度
        max_queue_len = max(len(self.queues[s].items) for s in WORKSTATIONS.keys())
        bottleneck_congestion = float(max_queue_len) / float(self._queue_len_norm)
        
        # 当前队列长度
        current_queue_len = len(self.queues[station_name].items)
        queue_len_normalized = float(current_queue_len) / float(self._queue_len_norm)
        
        global_features = np.array([
            time_normalized,
            wip_normalized,
            np.clip(bottleneck_congestion, 0, 1.0),
            np.clip(queue_len_normalized, 0, 1.0),
        ], dtype=np.float32)

        # 10-27-16-30 修复注释：当前队列摘要统计为 30维 = 6特征 × 5统计
        # --- 3. 当前队列摘要统计 (Queue Summary) - 30维 ---
        queue_summary = self._get_queue_summary_features(station_name)
        
        # --- 4. 候选工件详细特征 (Candidate Workpieces) - 90维 ---
        candidate_features = self._get_candidate_features(station_name)
        
        extra_features = None
        try:
            if bool(self.config.get('include_future_arrival_summary', True)):
                extra_features = self._get_future_arrival_summary_features(float(self.env.now))
        except Exception:
            extra_features = None
        
        # 组合所有特征
        if isinstance(extra_features, np.ndarray) and extra_features.size > 0:
            full_obs = np.concatenate([agent_features, global_features, queue_summary, candidate_features, extra_features])
        else:
            full_obs = np.concatenate([agent_features, global_features, queue_summary, candidate_features])
        return full_obs.flatten()

    def _get_future_arrival_summary_features(self, now: float) -> np.ndarray:
        _t = float(self._simulation_time) if float(self._simulation_time) > 0 else float(SIMULATION_TIME)
        total_parts_in_system = float(sum(o.quantity for o in self.orders))
        denom = total_parts_in_system if total_parts_in_system > 0 else 1.0

        next_dt = None
        parts_30 = 0.0
        parts_60 = 0.0
        parts_120 = 0.0
        for o in (self.orders or []):
            try:
                at = float(getattr(o, 'arrival_time', 0.0))
                dt = at - float(now)
                if dt <= 0.0:
                    continue
                q = float(getattr(o, 'quantity', 0.0))
                if next_dt is None or dt < next_dt:
                    next_dt = dt
                if dt <= 30.0:
                    parts_30 += q
                if dt <= 60.0:
                    parts_60 += q
                if dt <= 120.0:
                    parts_120 += q
            except Exception:
                continue

        if next_dt is None:
            next_dt = float(_t)

        next_dt_norm = float(np.clip(float(next_dt) / float(_t), 0.0, 1.0))
        p30 = float(np.clip(parts_30 / denom, 0.0, 1.0))
        p60 = float(np.clip(parts_60 / denom, 0.0, 1.0))
        p120 = float(np.clip(parts_120 / denom, 0.0, 1.0))
        return np.array([next_dt_norm, p30, p60, p120], dtype=np.float32)
    
    def _get_queue_summary_features(self, station_name: str) -> np.ndarray:
        """
        队列摘要统计特征 (30维 = 6特征 × 5统计量)
        保留：纯工艺负载特征
        """
        queue = self.queues[station_name].items
        
        if not queue:
            # 空队列返回零向量
            return np.zeros(30, dtype=np.float32)
        
        # 收集各种特征
        processing_times = []
        remaining_ops = []
        remaining_total_times = []
        downstream_congestions = []
        priorities = []
        is_final_ops = []
        
        for part in queue:
            processing_times.append(part.get_processing_time())
            
            route = get_route_for_product(part.product_type)
            remaining_ops_count = len(route) - part.current_step
            remaining_ops.append(remaining_ops_count)
            
            remaining_total_times.append(_calculate_part_total_remaining_processing_time(part))
            
            # 下游拥堵
            if part.current_step < len(route) - 1:
                downstream_station = route[part.current_step + 1]["station"]
                congestion = len(self.queues[downstream_station].items)
                downstream_congestions.append(congestion)
            else:
                downstream_congestions.append(0)
            
            priorities.append(part.priority)
            is_final_ops.append(1.0 if remaining_ops_count <= 1 else 0.0)
        
        # 计算5种统计量：min, max, mean, std, median
        def compute_stats(values):
            if not values:
                return [0.0, 0.0, 0.0, 0.0, 0.0]
            arr = np.array(values)
            return [
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.mean(arr)),
                float(np.std(arr)),
                float(np.median(arr)),
            ]
        
        # 归一化并收集统计
        features = []

        # 1. 加工时间统计
        proc_norm = [p / ENHANCED_OBS_CONFIG["max_op_duration_norm"] for p in processing_times]
        features.extend(compute_stats(proc_norm))
        
        # 2. 剩余工序统计
        ops_norm = [o / ENHANCED_OBS_CONFIG["max_bom_ops_norm"] for o in remaining_ops]
        features.extend(compute_stats(ops_norm))
        
        # 3. 剩余总时间统计
        time_norm = [t / ENHANCED_OBS_CONFIG["total_remaining_time_norm"] for t in remaining_total_times]
        features.extend(compute_stats(time_norm))
        
        # 4. 下游拥堵统计
        cong_norm = [float(c) / float(self._queue_len_norm) for c in downstream_congestions]
        features.extend(compute_stats(cong_norm))
        
        # 5. 优先级统计
        prio_norm = [p / 5.0 for p in priorities]
        features.extend(compute_stats(prio_norm))
        
        # 6. 最终工序标记统计
        features.extend(compute_stats(is_final_ops))
        
        return np.array(features, dtype=np.float32)
    
    def _get_candidate_features(self, station_name: str) -> np.ndarray:
        """
        # 10-21-22-30：更正维度注释
        # 方案B：获取候选工件详细特征 (90维 = 9维 × 10工件)
        采用多样性采样策略
        """
        candidates = self._get_candidate_workpieces(station_name)
        
        feature_list = []
        candidate_dim = ENHANCED_OBS_CONFIG["candidate_feature_dim"]
        
        for i in range(ENHANCED_OBS_CONFIG["num_candidate_workpieces"]):
            if i < len(candidates):
                part = candidates[i]['part']
                features = self._get_workpiece_obs(part, current_station=station_name)
            else:
                # 空槽位用零填充
                features = np.zeros(candidate_dim, dtype=np.float32)
            feature_list.append(features)
        
        return np.concatenate(feature_list)
    
    def _get_action_mask(self, station_name: str) -> np.ndarray:
        """
        🔧 新增：生成动作掩码，标记哪些动作是有效的
        
        掩码规则：
        - 动作0 (IDLE): 当队列非空且本步仍有可用并发容量时禁用；否则允许
        - 动作1-N (候选工件): 仅当候选工件存在且前置工序已完成时有效
        
        Returns:
            action_mask: 形状为 (action_space_size,) 的布尔数组，True表示有效动作
        """
        action_size = 1 + int(ENHANCED_OBS_CONFIG.get("num_candidate_workpieces", 0))
        action_mask = np.ones(action_size, dtype=np.bool_)
        
        # 计算当前站点是否具备可用并发容量
        capacity = WORKSTATIONS[station_name]['count']
        busy = self.equipment_status[station_name]['busy_count']
        available_capacity = max(0, capacity - busy)
        queue = self.queues[station_name].items

        # 收紧IDLE：当“有货可做且仍有可用并发容量”时，禁止IDLE
        # 其余情况下（无货或无可用并发或设备故障等待）允许IDLE
        action_mask[0] = not (len(queue) > 0 and available_capacity > 0)
        
        # 检查候选工件动作的有效性
        candidates = self._get_candidate_workpieces(station_name)
        
        for i in range(len(candidates)):
            action_idx = self._candidate_action_start + i
            if action_idx < action_size:
                candidate_info = candidates[i]
                part = candidate_info.get('part')
                
                # 检查零件是否存在且前置工序已完成
                if part is None:
                    action_mask[action_idx] = False
                else:
                    # 检查零件是否在当前队列中（可能已被处理）
                    part_in_queue = any(p.part_id == part.part_id for p in queue)
                    if not part_in_queue:
                        action_mask[action_idx] = False
                    else:
                        # 检查前置工序是否完成（零件是否在当前工作站）
                        current_station = part.get_current_station()
                        if current_station != station_name:
                            action_mask[action_idx] = False
                        else:
                            action_mask[action_idx] = True
        
        # 对于超出候选数量的动作，标记为无效
        for i in range(len(candidates), ENHANCED_OBS_CONFIG.get("num_candidate_workpieces", 0)):
            action_idx = self._candidate_action_start + i
            if action_idx < action_size:
                action_mask[action_idx] = False
        
        return action_mask
    
    def _get_candidate_workpieces(self, station_name: str) -> List[Dict[str, Any]]:
        """
        方案B：获取候选工件列表（多样性采样）
        
        核心思想：打破FIFO锁定，提供全局视野
        - 通过多样性采样确保agent能看到队列中不同类型的工件
        - 不再受限于队列前几个位置，实现真正的全局优化
        
        采样策略（恢复混合：紧急 + 最短 + 随机）：
        - 紧急EDD：按最小松弛度(负值更紧急)选取 num_urgent_candidates 个
        - 最短SPT：按当前工序加工时间从小到大选取 num_short_candidates 个
        - 随机Random：从剩余索引中选取 num_random_candidates 个
        - 总候选数不超过 ENHANCED_OBS_CONFIG["num_candidate_workpieces"]
        
        当 deterministic_candidates=True 时：
        - EDD/SPT 分支使用稳定排序后直接取前N个
        - Random 分支使用队列顺序取前N个（不随机）
        
        返回格式：[{"part": Part, "index": int, "category": str}, ...]
        
        10-24-21-50 恢复混合候选采样(EDD+SPT+随机)，并支持确定性评估复现
        """
        queue = self.queues[station_name].items
        
        if not queue:
            # 空队列清空缓存
            self._cached_candidates[station_name] = []
            return []
        
        # 若本步已有缓存，直接返回，确保观测与执行一致
        if station_name in self._cached_candidates and self._cached_candidates[station_name]:
            return self._cached_candidates[station_name]

        candidates: List[Dict[str, Any]] = []
        used_indices = set()

        # 10-24-21-50 读取配额（紧急/最短/随机）
        num_total = int(ENHANCED_OBS_CONFIG.get("num_candidate_workpieces", 0))
        num_urgent = int(ENHANCED_OBS_CONFIG.get("num_urgent_candidates", 0))
        num_short = int(ENHANCED_OBS_CONFIG.get("num_short_candidates", 0))
        num_random = int(ENHANCED_OBS_CONFIG.get("num_random_candidates", 0))
        # 若三者之和超过总量，进行裁剪
        quota_sum = num_urgent + num_short + num_random
        if quota_sum > num_total:
            # 10-24-21-50 保守裁剪：按比例下调，至少为0
            scale = num_total / max(1, quota_sum)
            num_urgent = int(num_urgent * scale)
            num_short = int(num_short * scale)
            num_random = max(0, num_total - num_urgent - num_short)

        available_indices = list(range(len(queue)))

        # 10-24-21-50 分支一：EDD（最小松弛度）
        if num_urgent > 0 and available_indices:
            # 计算每个索引的slack
            slack_list = []
            current_time = self.env.now
            for idx in available_indices:
                part = queue[idx]
                slack_val = calculate_slack_time(part, current_time, self.queues, WORKSTATIONS)
                slack_list.append((idx, slack_val, part.part_id))
            # 稳定排序：slack升序，part_id次序保证稳定
            slack_list.sort(key=lambda x: (x[1], x[2]))
            urgent_indices = [t[0] for t in slack_list[:min(num_urgent, len(slack_list))]] if self._deterministic_candidates else [t[0] for t in slack_list[:min(num_urgent, len(slack_list))]]
            for idx in urgent_indices:
                candidates.append({"part": queue[idx], "index": idx, "category": "urgent"})
                used_indices.add(idx)

        # 10-24-21-50 分支二：SPT（当前工序时间最短）
        if num_short > 0 and len(used_indices) < len(available_indices):
            rem_indices = [i for i in available_indices if i not in used_indices]
            spt_list = []
            for idx in rem_indices:
                part = queue[idx]
                proc = float(part.get_processing_time())
                spt_list.append((idx, proc, part.part_id))
            spt_list.sort(key=lambda x: (x[1], x[2]))
            short_indices = [t[0] for t in spt_list[:min(num_short, len(spt_list))]]
            for idx in short_indices:
                candidates.append({"part": queue[idx], "index": idx, "category": "short"})
                used_indices.add(idx)

        # 10-24-21-50 分支三：随机（或确定性顺序）
        if num_random > 0 and len(used_indices) < len(available_indices):
            rem_indices = [i for i in available_indices if i not in used_indices]
            if rem_indices:
                sample_size = min(num_random, len(rem_indices))
                if self._deterministic_candidates:
                    sampled_indices = rem_indices[:sample_size]
                else:
                    # 10-23-16-05 稳定哈希种子，确保跨进程/运行可复现
                    import hashlib
                    seed_tuple = (station_name, int(self.env.now), tuple(p.part_id for p in queue), "random")
                    h = hashlib.sha256(str(seed_tuple).encode('utf-8')).hexdigest()
                    seed = int(h[:8], 16)
                    rng = random.Random(seed)
                    sampled_indices = rng.sample(rem_indices, sample_size)
                for idx in sampled_indices:
                    candidates.append({"part": queue[idx], "index": idx, "category": "random"})
                    used_indices.add(idx)

        # 10-24-21-50 若仍不足总候选配额，补齐（按队列顺序或剩余随机）
        if len(candidates) < num_total:
            rem_indices = [i for i in available_indices if i not in used_indices]
            if rem_indices:
                need = num_total - len(candidates)
                if self._deterministic_candidates:
                    fill_indices = rem_indices[:need]
                else:
                    import hashlib
                    seed_tuple = (station_name, int(self.env.now), tuple(p.part_id for p in queue), "fill")
                    h = hashlib.sha256(str(seed_tuple).encode('utf-8')).hexdigest()
                    seed = int(h[:8], 16)
                    rng = random.Random(seed)
                    if need >= len(rem_indices):
                        fill_indices = rem_indices
                    else:
                        fill_indices = rng.sample(rem_indices, need)
                for idx in fill_indices:
                    candidates.append({"part": queue[idx], "index": idx, "category": "random"})
                    used_indices.add(idx)

        # 缓存本步候选以保证一致性
        self._cached_candidates[station_name] = candidates
        return candidates
    
    def _select_workpiece_by_action(self, station_name: str, action: int) -> Optional[Tuple[Part, int]]:
        """
        方案A：纯候选动作选择工件
        

        - 智能体必须从多样性候选工件中学习选择
        - 不再依赖EDD、SPT等经过验证的算法
        - 通过候选工件的多样性采样，提供充分的学习材料
        
        动作映射：
        - 0: IDLE（不处理）
        - 1-10: 候选工件1-10（从多样性采样列表中选择）
        
        返回：(选中的工件, 在队列中的索引) 或 None
        """
        queue = self.queues[station_name].items
        
        if not queue or action == 0:
            return None
        
        # 候选工件动作 (1-10)
        if self._candidate_action_start <= action <= self._candidate_action_end:
            candidates = self._get_candidate_workpieces(station_name)
            candidate_idx = action - self._candidate_action_start
            if candidate_idx < len(candidates):
                candidate_info = candidates[candidate_idx]
                part = candidate_info['part']
                
                # 需要找到这个工件在当前队列中的实际索引
                # 🔧 核心修复：增加 part is not None 的检查，防止选择到已处理的候选槽
                if part:
                    for idx, queue_part in enumerate(queue):
                        if queue_part.part_id == part.part_id:
                            return (part, idx)
        
        return None
    
    def _get_workpiece_obs(self, part: Part, current_station: str = None) -> np.ndarray:
        """
        保留：纯中性的工艺和负载特征（8维）
        """
        # 特征1: 是否存在
        exists = 1.0
        
        # 特征2: 剩余工序数
        route = get_route_for_product(part.product_type)
        remaining_ops = len(route) - part.current_step
        normalized_remaining_ops = remaining_ops / ENHANCED_OBS_CONFIG["max_bom_ops_norm"]
        
        # 特征3: 剩余总加工时间
        total_remaining_time = _calculate_part_total_remaining_processing_time(part)
        normalized_total_remaining_time = total_remaining_time / ENHANCED_OBS_CONFIG["total_remaining_time_norm"]

        # 特征4: 当前工序加工时间
        current_op_duration = part.get_processing_time()
        normalized_op_duration = current_op_duration / ENHANCED_OBS_CONFIG["max_op_duration_norm"]
        
        # 特征5: 下游拥堵情况
        downstream_congestion = 0.0
        if part.current_step < len(route) - 1:
            downstream_station = route[part.current_step + 1]["station"]
            if downstream_station in self.queues:
                congestion = len(self.queues[downstream_station].items)
                downstream_congestion = np.clip(congestion / ENHANCED_OBS_CONFIG["w_station_capacity_norm"], 0, 1.0)
        
        # 特征6: 订单优先级
        priority = part.priority / 5.0

        # 特征7: 是否为最终工序
        is_final_op = 1.0 if remaining_ops <= 1 else 0.0
        
        # 特征8: 产品类型编码（简化为产品ID）
        product_id = 0.0
        try:
            stable_products = list(SYSTEM_PRODUCT_TYPES)
            unknown_idx = len(stable_products)
            denom = float(max(1, unknown_idx + 1))
            if part.product_type in stable_products:
                product_id = float(stable_products.index(part.product_type)) / denom
            else:
                product_id = float(unknown_idx) / denom
        except Exception:
            product_id = 0.0
        
        # V2新增特征9: 时间压力感知（基于物理时间关系）
        # 计算逻辑：压力 = 剩余加工时间 / (距离交期的剩余时间 + 1.0)
        # 压力值越大表示时间越紧张，≥1.0表示已无法按时完成
        remaining_time_to_due = part.due_date - self.env.now
        if remaining_time_to_due > 0:
            time_pressure = total_remaining_time / (remaining_time_to_due + 1.0)
        else:
            # 已超期：压力值设为最大
            time_pressure = 2.0
        time_pressure_normalized = np.clip(time_pressure / 2.0, 0, 1.0)  # 归一化到[0,1]

        # 🔧 新增：真实slack（分钟）：due_date - (current_time + remaining_total_time)
        slack = float(part.due_date) - float(self.env.now) - float(total_remaining_time)
        slack_norm = float(REWARD_CONFIG.get('slack_tardiness_normalize_scale', 480.0))
        slack_normalized = float(np.clip(slack / (slack_norm if slack_norm > 0 else 1.0), -1.0, 1.0))

        # 10-23-14-50 新增：压缩归一化，缓解跨阶段/随机订单的饱和
        if ENHANCED_OBS_CONFIG.get("use_compressed_norm", False):
            def _compress(x: float) -> float:
                return float(x) / (1.0 + float(x)) if x >= 0 else 0.0
            normalized_total_remaining_time = _compress(np.clip(normalized_total_remaining_time, 0, 10.0))
            normalized_op_duration = _compress(np.clip(normalized_op_duration, 0, 10.0))
            # downstream_congestion/priority/is_final_op/product_id本身在[0,1]
            time_pressure_normalized = _compress(time_pressure_normalized)

        feature_list = [
            exists,
            np.clip(normalized_remaining_ops, 0, 1.0),
            np.clip(normalized_total_remaining_time, 0, 1.0),
            np.clip(normalized_op_duration, 0, 1.0),
            downstream_congestion,
            priority,
            is_final_op,
            product_id,
            time_pressure_normalized,
            slack_normalized,
        ]
        
        return np.array(feature_list, dtype=np.float32)


    def get_global_state(self) -> np.ndarray:
        """🔧 MAPPO关键修复：获取真正的全局状态，包含环境全局信息而非局部观察拼接"""
        global_features = []
        
        # 1. 环境时间信息
        _t = float(self._simulation_time) if float(self._simulation_time) > 0 else float(SIMULATION_TIME)
        time_normalized = self.env.now / _t
        global_features.append(time_normalized)
        
        # 2. 全局任务进度
        total_parts_in_system = sum(order.quantity for order in self.orders)
        completed_parts_ratio = len(self.completed_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        active_parts_ratio = len(self.active_parts) / total_parts_in_system if total_parts_in_system > 0 else 0.0
        global_features.extend([completed_parts_ratio, active_parts_ratio])
        
        # 3. 所有工作站的汇总状态（顺序与agents一致）
        for station_name in WORKSTATIONS.keys():
            # 队列长度归一化
            queue_len = len(self.queues[station_name].items)
            denom = float(getattr(self, "_queue_len_norm", 1.0))
            if denom <= 0:
                denom = 1.0
            queue_len_norm = float(queue_len) / denom
            global_features.append(np.clip(queue_len_norm, 0.0, 1.0))
            
            # 设备忙碌率
            capacity = WORKSTATIONS[station_name]['count']
            if capacity and capacity > 0:
                busy_ratio = self.equipment_status[station_name]['busy_count'] / capacity
            else:
                busy_ratio = 0.0
            global_features.append(busy_ratio)
            
            # 设备故障状态
            is_failed = 1.0 if self.equipment_status[station_name]['is_failed'] else 0.0
            global_features.append(is_failed)
        
        # 5. 全局KPI趋势（修复：使用累积利用率而非瞬时值）
        cumulative_utilizations = []
        for station_name in WORKSTATIONS.keys():
            status = self.equipment_status[station_name]
            capacity = WORKSTATIONS[station_name]['count']
            
            # 专家修复 V3.1：修正错误的属性访问，应为 part.contribution_map
            if self.env.now > status.get('last_event_time', 0.0):
                elapsed = self.env.now - status.get('last_event_time', 0.0)
                busy_count = status.get('busy_count', 0)
                # 这个更新是临时的，不会写回status字典，仅用于计算当前全局状态
                current_busy_machine_time = status.get('busy_machine_time', 0.0) + elapsed * busy_count
            else:
                current_busy_machine_time = status.get('busy_machine_time', 0.0)

            if self.env.now > 0 and capacity > 0:
                utilization = current_busy_machine_time / (self.env.now * capacity)
                cumulative_utilizations.append(np.clip(utilization, 0.0, 1.0))
            else:
                cumulative_utilizations.append(0.0)
                
        avg_cumulative_utilization = np.mean(cumulative_utilizations) if cumulative_utilizations else 0.0
        
        global_features.append(avg_cumulative_utilization)
        
        return np.array(global_features, dtype=np.float32)

    def step_with_actions(self, actions: Dict[str, int]) -> Dict[str, float]:
        """
        执行一步仿真，支持并行的MultiDiscrete动作
        """
        # 记录执行前状态
        prev_completed = len(self.completed_parts)
        prev_total_steps = sum(part.current_step for part in self.active_parts)
        
        # 用于防止同一个工件在一个step内被多次选择
        selected_part_ids_this_step = set()
        # 本步内各站点已启动的零件计数（防止超过本步并发能力）
        local_start_count = defaultdict(int)

        # 执行智能体动作
        actions_executed = 0
        decision_time = self.env.now
        action_context: Dict[str, Dict[str, Any]] = {}

        for agent_id, agent_action in actions.items():
            station_name = agent_id.replace("agent_", "")
            pre_queue_snapshot = list(self.queues[station_name].items)
            
            # 确保 agent_action 是可迭代的 (MultiDiscrete返回数组，Discrete返回标量)
            if not isinstance(agent_action, (list, np.ndarray)):
                agent_action = [agent_action]

            try:
                capacity = int(WORKSTATIONS[station_name]['count'])
            except Exception:
                capacity = 0
            try:
                busy_count = int(self.equipment_status[station_name]['busy_count'])
            except Exception:
                busy_count = 0
            available_capacity = max(0, capacity - busy_count)

            context = {
                "queue_len_before": len(pre_queue_snapshot),
                "queue_snapshot": pre_queue_snapshot,
                "decision_time": decision_time,
                "action": agent_action,
                "available_capacity": available_capacity,
                "selected_part": None,
                "processed": False,
                "started_parts": [],  # 记录本步该agent启动的所有零件及其决策时slack
                # 10-21-22-45 修复：统计无效/冲突动作尝试次数（即便成功回退也记惩罚）
                "invalid_attempts": 0
            }
            action_context[agent_id] = context

            # --- 阶段一：决策与锁定 (Lock Phase) ---
            # 基于决策时刻的统一状态，为该智能体的所有并行设备（机器）选择工件
            parts_to_process_this_agent: List[Part] = []

            # 遍历该智能体的每一个动作（对应每一台机器）
            for machine_action in agent_action:
                if machine_action > 0:
                    # 检查真实可用容量（考虑本步已为该站点锁定的零件）
                    already_started_this_step = local_start_count.get(station_name, 0)
                    real_available_capacity = max(0, 
                        WORKSTATIONS[station_name]['count'] - 
                        self.equipment_status[station_name]['busy_count'] - 
                        already_started_this_step
                    )
                    
                    if real_available_capacity > 0:
                        result = self._select_workpiece_by_action(station_name, machine_action)
                        if result is not None:
                            selected_part, part_index = result
                            
                            # 锁定工件：加入待处理列表
                            parts_to_process_this_agent.append(selected_part)
                            
                            # 全局去重：将part_id加入全局已选集合
                            selected_part_ids_this_step.add(selected_part.part_id)
                            
                            # 更新本站点的本地计数器，用于计算下一台机器的可用容量
                            local_start_count[station_name] += 1
                            
                            # 记录启动的零件及其决策时的slack，用于奖励计算
                            context["started_parts"].append({
                                "part_id": selected_part.part_id,
                                "slack": calculate_slack_time(selected_part, decision_time, self.queues, WORKSTATIONS)
                            })
                        else:
                            context["invalid_attempts"] = context.get("invalid_attempts", 0) + 1
                    else:
                        # 10-27-17-30 新增：容量不足时对非零动作记录轻微惩罚，减少多头冗余动作对梯度的噪声
                        context["invalid_attempts"] = context.get("invalid_attempts", 0) + 1

            # --- 阶段二：执行 (Execute Phase) ---
            # 在所有决策完成后，统一处理本智能体已锁定的所有工件
            if parts_to_process_this_agent:
                context["processed"] = True
                actions_executed += len(parts_to_process_this_agent)
                
                for part_to_process in parts_to_process_this_agent:
                    # 此处才从队列中移除工件，并启动simpy处理进程
                    self._process_part_at_station(station_name, part_to_process=part_to_process)

        # 推进仿真
        try:
            self.env.run(until=self.env.now + 1)
        except simpy.core.EmptySchedule:
            self.simulation_ended = True
        
        self.current_time = self.env.now
        
        # 计算奖励
        rewards = self.get_rewards(actions, action_context)

        # 本步结束后清空候选缓存（下一步将重建）
        self._cached_candidates.clear()
        
        # 训练模式下完全静默调试信息
        if not self._training_mode and self.debug_level == 'DEBUG':
            new_completed = len(self.completed_parts)
            new_total_steps = sum(part.current_step for part in self.active_parts)
            
            if new_completed > prev_completed or new_total_steps > prev_total_steps:
                print(f"🎯 进度更新: 完成零件 {prev_completed}->{new_completed}, 总工序 {prev_total_steps}->{new_total_steps}")
                print(f"   执行动作数: {actions_executed}, 奖励: {list(rewards.values())}")
        
        return rewards
    
    def _process_part_at_station(self, station_name: str, part_to_process: Part = None, part_index: int = 0):
        """
        在指定工作站处理零件 - 增强版
        - 可以选择处理队列中的特定零件 (通过part_to_process或part_index)
        """
        part = None
        actual_part_index = -1

        if part_to_process:
            # 优先使用part对象定位
            for i, p in enumerate(self.queues[station_name].items):
                if p.part_id == part_to_process.part_id:
                    part = p
                    actual_part_index = i
                    break
        elif part_index < len(self.queues[station_name].items):
             # 后备方案：使用索引
            part = self.queues[station_name].items[part_index]
            actual_part_index = part_index

        if not part or actual_part_index == -1:
            return # 零件未找到或索引越界
            
        # 检查设备是否可用
        if self.equipment_status[station_name]['busy_count'] < WORKSTATIONS[station_name]['count']:
            # 从队列中移除零件
            self.queues[station_name].items.pop(actual_part_index)
            
            # 启动处理进程
            self.env.process(self._execute_processing(station_name, part))
    
    def _execute_processing(self, station_name: str, part: Part):
        """执行零件加工"""
        # 请求设备资源
        with self.resources[station_name].request() as request:
            yield request
            # 若设备当前处于故障，等待修复结束
            status = self.equipment_status.get(station_name, {})
            if status.get('is_failed', False):
                repair_end = status.get('failure_end_time', self.env.now)
                wait_time = max(0.0, repair_end - self.env.now)
                if wait_time > 0:
                    yield self.env.timeout(wait_time)
            
            # 更新设备状态
            self._update_equipment_status(station_name, busy=True)
            
            # 执行加工
            start_time = self.env.now
            processing_time = part.get_processing_time()
            yield self.env.timeout(processing_time)
            finish_time = self.env.now
            
            # 新增：记录加工历史用于生成甘特图
            self.gantt_chart_history.append({
                "Task": f"Part-{part.part_id}",
                "Start": start_time,
                "Finish": finish_time,
                "Duration": finish_time - start_time,
                "Resource": station_name,
                "Product": part.product_type,
                "Part ID": part.part_id,
                "Order ID": part.order_id
            })
            
            # 更新设备状态
            self._update_equipment_status(station_name, busy=False)
            
            # 专家修复 V3.1：修正错误的属性访问，应为 part.contribution_map
            part.contribution_map[station_name] = part.contribution_map.get(station_name, 0.0) + processing_time
            
            # 零件完成当前工序
            part.current_step += 1
            
            # 检查是否完成所有工序
            if part.is_completed():
                part.completion_time = self.env.now
                self.completed_parts.append(part)
                # 🔧 关键修复：从活跃零件列表中移除完成的零件
                if part in self.active_parts:
                    self.active_parts.remove(part)
                self._update_completion_stats(part)
            else:
                # 移动到下一个工作站
                next_station = part.get_current_station()
                if next_station:
                    yield self.queues[next_station].put(part)

    def _calculate_progress_shaping_reward(self) -> float:
        """基于整体工序完成进度的塑形奖励"""
        if not self.orders:
            return 0.0
        
        # 计算总工序完成率
        total_steps_done = sum(part.current_step for part in self.active_parts)
        total_steps_done += sum(
            len(get_route_for_product(part.product_type)) 
            for part in self.completed_parts
        )
        
        max_possible_steps = sum(
            len(get_route_for_product(order.product)) * order.quantity
            for order in self.orders
        )
        
        if max_possible_steps == 0:
            return 0.0
        
        progress_ratio = total_steps_done / max_possible_steps
        
        # 进度增量奖励
        progress_delta = progress_ratio - self._last_progress_ratio
        self._last_progress_ratio = progress_ratio
        
        # 归一化并分配到每个agent
        shaping_reward = REWARD_CONFIG["progress_shaping_coeff"] * progress_delta / len(WORKSTATIONS)
        return shaping_reward
    
    def _calculate_urgency_reduction_reward(self) -> float:
        """基于紧急度降低的引导奖励（替代原密集奖励）"""
        if not self.active_parts:
            current_urgency = 0.0
        else:
            # 计算当前紧急度（使用更稳定的指标）
            current_urgency = 0.0
            for part in self.active_parts:
                remaining_time = part.due_date - self.env.now
                remaining_processing = _calculate_part_total_remaining_processing_time(part)
                # 紧急度 = max(0, 需要的时间 - 剩余的时间)
                urgency = max(0, remaining_processing - remaining_time)
                current_urgency += urgency
        
        # 紧急度降低 = 正奖励
        urgency_delta = self._last_urgency_sum - current_urgency
        self._last_urgency_sum = current_urgency
        
        # 归一化并分配
        reward = REWARD_CONFIG["urgency_reduction_reward"] * (urgency_delta / 480.0) / len(WORKSTATIONS)
        return reward
    
    def get_rewards(self, actions: Dict[str, int], action_context: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """分层奖励系统：完成率 > 时间质量 > 过程塑形"""
        rewards = {f"agent_{station}": 0.0 for station in WORKSTATIONS.keys()}
        
        # ============================================================
        # 第一层：任务完成奖励（主导）
        # ============================================================
        current_completed = len(self.completed_parts)
        new_completed_parts_count = current_completed - self.stats.get('last_completed_count', 0)
        self.stats['last_completed_count'] = current_completed
        
        if new_completed_parts_count > 0:
            recent_completed = self.completed_parts[-new_completed_parts_count:]
            
            for part in recent_completed:
                # 基础完成奖励（无论延期与否）
                base_reward = REWARD_CONFIG["part_completion_reward"]
                
                # 时间质量调整
                tardiness = max(0.0, part.completion_time - part.due_date)
                if tardiness == 0:
                    # 按时完成：额外奖励
                    time_reward = base_reward + REWARD_CONFIG["on_time_completion_reward"]
                else:
                    # 延期完成：基础奖励 - 稳健化（Huber）迟期惩罚（基于归一化 tardiness）
                    tardiness_norm = float(tardiness / 480.0)
                    if REWARD_CONFIG.get("use_huber_tardiness", False):
                        delta = float(REWARD_CONFIG.get("tardiness_huber_delta_norm", 0.3))
                        ax = abs(tardiness_norm)
                        if ax <= delta:
                            huber_val = 0.5 * (tardiness_norm ** 2)
                        else:
                            huber_val = delta * (ax - 0.5 * delta)
                    else:
                        huber_val = tardiness_norm
                    tardiness_penalty = REWARD_CONFIG["tardiness_penalty_scaler"] * huber_val
                    time_reward = base_reward + tardiness_penalty  # 负值为惩罚
                
                # 10202115 按各站点对该零件的累计加工时间占比进行奖励分配（替代均分）
                if part.contribution_map:
                    total_contribution = sum(part.contribution_map.values())
                    if total_contribution > 0:
                        for station_name, contribution in part.contribution_map.items():
                            agent_id = f"agent_{station_name}"
                            if agent_id in rewards:
                                weight = contribution / total_contribution
                                rewards[agent_id] += time_reward * weight
        
        # 终局全部完成奖励
        if self.is_done():
            total_required = sum(order.quantity for order in self.orders)
            if len(self.completed_parts) >= total_required:
                if not self.final_bonus_awarded:
                    final_bonus = REWARD_CONFIG["final_all_parts_completion_bonus"]
                    for agent_id in rewards:
                        rewards[agent_id] += final_bonus
                    self.final_bonus_awarded = True
                    self.final_bonus_value = final_bonus * len(rewards)
        
        # ============================================================
        # 第二层：过程塑形奖励（辅助）
        # ============================================================
        # 2.1 进度塑形（基于工序完成率）
        progress_reward = self._calculate_progress_shaping_reward()
        for agent_id in rewards:
            rewards[agent_id] += progress_reward
        
        # 2.2 行为约束（最小化）
        for agent_id, action in actions.items():
            context = action_context.get(agent_id, {})
            queue_len_before = context.get("queue_len_before", 0)
            
            # 统一动作判定，兼容 MultiDiscrete（数组）与 Discrete（标量）
            if isinstance(action, (list, np.ndarray)):
                action_arr = np.array(action)
                any_positive = np.any(action_arr > 0)
                all_zero = np.all(action_arr == 0)
            else:
                any_positive = (action > 0)
                all_zero = (action == 0)

            # 若有非零动作但未成功启动任何零件，则视为无效动作
            started_parts = context.get("started_parts", [])
            if any_positive and len(started_parts) == 0:
                rewards[agent_id] += REWARD_CONFIG["invalid_action_penalty"]
            
            # 若全部为零且队列非空，判定为不必要的空转
            available_capacity = int(context.get("available_capacity", 0))
            if all_zero and queue_len_before > 0 and available_capacity > 0:
                rewards[agent_id] += REWARD_CONFIG["unnecessary_idle_penalty"]

                idle_wa_pen = float(REWARD_CONFIG.get("idle_when_work_available_penalty", 0.0))
                if idle_wa_pen != 0.0:
                    rewards[agent_id] += idle_wa_pen
                    self.stats['idle_when_work_available_count'] = int(self.stats.get('idle_when_work_available_count', 0)) + 1

            # 10-21-22-45 修复：对被回退机制“修正”的无效/冲突尝试也进行惩罚，避免环境替代学习信号
            invalid_attempts = int(context.get("invalid_attempts", 0))
            if invalid_attempts > 0:
                rewards[agent_id] += REWARD_CONFIG["invalid_action_penalty"] * float(invalid_attempts)
        
        # 2.3 紧急度引导（替代密集奖励）
        urgency_reward = self._calculate_urgency_reduction_reward()
        for agent_id in rewards:
            rewards[agent_id] += urgency_reward
        
        # 2.4 (核心改进) 基于负松弛时间的持续惩罚
        # 这个惩罚是即时的、密集的，且与延期的严重程度成正比。
        # 它会迫使智能体优先处理最紧急（负松弛时间最大）的工件，从而学会管理延期。
        slack_penalty_coeff = REWARD_CONFIG.get("slack_time_penalty_coeff", 0.0)
        if slack_penalty_coeff != 0.0:
            tanh_scale = float(REWARD_CONFIG.get("slack_penalty_tanh_scale", 240.0))
            max_abs_penalty = float(REWARD_CONFIG.get("slack_penalty_max_abs", 50.0))
            eps = 1e-6
            for station_name in WORKSTATIONS.keys():
                agent_id = f"agent_{station_name}"
                total_negative_slack_in_queue = 0.0
                
                # 遍历该工作站队列中的每一个工件
                for part in self.queues[station_name].items:
                    slack = calculate_slack_time(part, self.env.now)
                    if slack < 0:
                        total_negative_slack_in_queue += float(abs(slack))
                
                if total_negative_slack_in_queue > 0:
                    # 使用tanh缩放以限制极端值，并对单步单agent惩罚做绝对上限裁剪
                    scaled = np.tanh(total_negative_slack_in_queue / (tanh_scale + eps)) * tanh_scale
                    penalty = float(slack_penalty_coeff) * float(scaled)
                    # 绝对值裁剪
                    if penalty > max_abs_penalty:
                        penalty = max_abs_penalty
                    if penalty < -max_abs_penalty:
                        penalty = -max_abs_penalty
                    rewards[agent_id] += penalty
        
        # ============================================================
        # 🔧 新增：基于Slack的非线性迟交惩罚（奖励函数重塑）
        # ============================================================
        if REWARD_CONFIG.get("slack_based_tardiness_enabled", True):
            normalize_scale = REWARD_CONFIG.get("slack_tardiness_normalize_scale", 480.0)
            threshold = REWARD_CONFIG.get("slack_tardiness_threshold", 0.0)
            beta_tard_step = REWARD_CONFIG.get("slack_tardiness_step_penalty", -0.5)
            gamma_overdue = REWARD_CONFIG.get("slack_tardiness_overdue_penalty", -2.0)
            zeta_wip = REWARD_CONFIG.get("wip_penalty_coeff", -0.01)
            eta_idle = REWARD_CONFIG.get("idle_penalty_coeff", -0.005)
            
            current_time = self.env.now
            total_wip = len(self.active_parts)
            
            for station_name in WORKSTATIONS.keys():
                agent_id = f"agent_{station_name}"
                queue = self.queues[station_name].items
                station_wip = len(queue)
                
                # 计算该站点的总负松弛时间和已迟交增量
                total_negative_slack = 0.0
                overdue_delta = 0.0
                
                for part in queue:
                    slack = calculate_slack_time(part, current_time)
                    remaining_proc_time = _calculate_part_total_remaining_processing_time(part)
                    
                    # 即将迟交的惩罚（负松弛时间）
                    if slack < threshold:
                        # 归一化负松弛时间
                        negative_slack_norm = abs(slack) / normalize_scale
                        total_negative_slack += negative_slack_norm
                    
                    # 已迟交的增量惩罚（如果零件已经完成但延期）
                    if part.completion_time is not None:
                        overdue = max(0.0, part.completion_time - part.due_date)
                        if overdue > 0:
                            overdue_norm = overdue / normalize_scale
                            # 使用Huber损失避免极端值
                            delta = 0.3
                            ax = abs(overdue_norm)
                            if ax <= delta:
                                huber_overdue = 0.5 * (overdue_norm ** 2)
                            else:
                                huber_overdue = delta * (ax - 0.5 * delta)
                            overdue_delta += huber_overdue
                
                # 应用惩罚（归一化到[-1, 1]范围）
                if total_negative_slack > 0:
                    slack_penalty = beta_tard_step * min(total_negative_slack, 1.0)  # 限制在[-1, 0]
                    rewards[agent_id] += slack_penalty
                
                if overdue_delta > 0:
                    overdue_penalty = gamma_overdue * min(overdue_delta, 1.0)  # 限制在[-2, 0]
                    rewards[agent_id] += overdue_penalty
                
                # WIP拥塞惩罚（归一化）
                wip_penalty = zeta_wip * min(station_wip / 20.0, 1.0)  # 假设最大WIP为20
                rewards[agent_id] += wip_penalty
                
                # 瓶颈闲置惩罚（如果队列非空但资源空闲）
                resource = self.resources.get(station_name)
                if resource and queue:
                    available_count = resource.count - len(resource.users)
                    if available_count > 0:
                        idle_penalty = eta_idle * (available_count / resource.count)
                        rewards[agent_id] += idle_penalty
        
        # ============================================================
        # 🔧 新增：评分分解项的逐步奖励塑形
        # ============================================================
        if REWARD_CONFIG.get('score_decomposition_shaping_enabled', False):
            try:
                weights = REWARD_CONFIG.get('score_decomposition_shaping_weights', {}) or {}
                potential_now = float(self._compute_score_decomposition_potential(weights))
                delta = potential_now - float(self._last_score_potential)
                self._last_score_potential = potential_now

                scale = float(REWARD_CONFIG.get('score_decomposition_shaping_scale', 0.0))
                clip_abs = float(REWARD_CONFIG.get('score_decomposition_shaping_clip_abs', 0.0))
                shaping = float(scale) * float(delta)

                if clip_abs > 0.0:
                    if shaping > clip_abs:
                        shaping = clip_abs
                    elif shaping < -clip_abs:
                        shaping = -clip_abs

                if shaping != 0.0:
                    per_agent = float(shaping) / float(max(1, len(rewards)))
                    for agent_id in rewards:
                        rewards[agent_id] += per_agent
            except Exception:
                pass
        
        # 更新订单进度与统计
        self._update_order_progress()
        return rewards
    
    def is_done(self) -> bool:
        """检查仿真是否结束 - 优先任务完成，时间作为备用条件"""
        # 优先检查任务完成，而不是时间耗尽
        
        # 条件1: 所有订单完成 (主要完成条件)
        total_required = sum(order.quantity for order in self.orders)
        if len(self.completed_parts) >= total_required:
            if not hasattr(self, '_completion_logged'):
                # 训练模式下完全静默
                if not SILENT_MODE and not self._training_mode:
                    print(f"🎉 所有订单完成! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._completion_logged = True
            return True
        
        # 条件2: 手动结束仿真
        if self.simulation_ended:
            return True
        
        # 条件3: 时间耗尽 (备用条件，增加时间限制)
        # 给智能体更多时间完成任务，避免总是超时截断
        max_time = float(self._max_sim_time)
        if self.current_time >= max_time:
            if not hasattr(self, '_timeout_logged'):
                # 训练模式下完全静默
                if not SILENT_MODE and not self._training_mode:
                    print(f"⏰ 时间耗尽! 完成{len(self.completed_parts)}/{total_required}个零件，用时{self.current_time:.1f}分钟")
                self._timeout_logged = True
            return True
        
        return False
    
    def get_final_stats(self) -> Dict[str, Any]:
        """获取最终统计结果，修复设备利用率计算异常"""
        # 关键修复：强制结算所有设备的最终忙碌时间
        for station_name, status in self.equipment_status.items():
            # 结算从 last_event_time 到当前时间的忙碌面积
            if self.current_time > status.get('last_event_time', 0.0):
                elapsed = self.current_time - status.get('last_event_time', 0.0)
                busy_count = status.get('busy_count', 0)
                # 这个更新是临时的，不会写回status字典，仅用于计算当前全局状态
                current_busy_machine_time = status.get('busy_machine_time', 0.0) + elapsed * busy_count
            else:
                current_busy_machine_time = status.get('busy_machine_time', 0.0)

            # 计算该工作站的设备利用率
            capacity = WORKSTATIONS[station_name]['count']
            if self.current_time > 0 and capacity > 0:
                utilization = current_busy_machine_time / (self.current_time * capacity)
            else:
                utilization = 0.0
            self.stats['equipment_utilization'][station_name] = utilization
        
        # 更可靠的平均利用率计算
        util_values = list(self.stats['equipment_utilization'].values())
        if util_values:
            mean_utilization = float(np.mean(util_values))
                # 移除调试信息，保持训练日志简洁
            if mean_utilization < 0.001 and len(self.completed_parts) > 0:
                # 静默处理异常情况，避免日志冗余
                pass
        else:
            mean_utilization = 0.0
        
        # 订单级统计延期
        total_tardiness = 0
        late_orders_count = 0

        for order in self.orders:
            if order.order_id in self.order_completion_times:
                # 订单已完成
                completion_time = self.order_completion_times[order.order_id]
                if completion_time > order.due_date:
                    tardiness = completion_time - order.due_date
                    total_tardiness += tardiness
                    late_orders_count += 1
            else:
                # 订单未完成，延期时间从交期算到仿真结束
                tardiness = max(0, self.current_time - order.due_date)
                total_tardiness += tardiness
                if tardiness > 0:
                    late_orders_count += 1
        
        total_required = sum(order.quantity for order in self.orders)
        
        if len(self.completed_parts) == total_required:
            # 所有零件都完成了，makespan是最后一个零件的完成时间
            if self.completed_parts:
                makespan = max(part.completion_time for part in self.completed_parts if part.completion_time is not None)
            else:
                makespan = self.current_time
        else:
            # 关键修复：未完成所有零件时，显示最后完成零件的时间
            if self.completed_parts:
                # 如果有零件完成，显示最后完成零件的时间
                makespan = max(part.completion_time for part in self.completed_parts if part.completion_time is not None)
            else:
                # 关键：如果没有零件完成，则将makespan设为当前耗尽的时间
                makespan = self.current_time
            self.stats['timeout_occurred'] = True
            self.stats['incomplete_parts'] = total_required - len(self.completed_parts)
        
        # 更新统计字段
        self.stats['mean_utilization'] = mean_utilization
        self.stats['total_tardiness'] = total_tardiness
        self.stats['total_parts'] = len(self.completed_parts)
        self.stats['makespan'] = makespan
        
        # 新增：写入事件时间线（保证可JSON序列化）
        try:
            self.stats['event_timeline'] = list(self.event_timeline)
        except Exception:
            self.stats['event_timeline'] = []
        
        return self.stats

# =============================================================================
# 3. PettingZoo多智能体环境接口 (PettingZoo Multi-Agent Environment)
# =============================================================================

class WFactoryEnv(ParallelEnv):
    """W工厂多智能体强化学习环境 - 基于PettingZoo"""
    
    metadata = {
        "render_modes": ["human"],
        "name": "w_factory_v1",
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config if config else {}
        self.sim = WFactorySim(self.config)
        self.agents = self.sim.agents
        self.possible_agents = self.sim.agents
        # --- 动作空间一致性断言：基于候选数量动态校验 ---
        _num_candidates = int(ENHANCED_OBS_CONFIG.get("num_candidate_workpieces", 0))
        _expected_action_space_size = 1 + _num_candidates  # 0=IDLE, 1-N=CANDIDATE_1~N
        _configured_action_space_size = ACTION_CONFIG_ENHANCED.get("action_space_size", _expected_action_space_size)
        if _configured_action_space_size != _expected_action_space_size:
            raise ValueError(
                f"动作空间大小配置不一致: 配置为{_configured_action_space_size}, 但根据候选数应为{_expected_action_space_size} (1 + num_candidate_workpieces)"
            )
        
        # 🔧 MAPPO修复：重新设计全局状态空间
        self._setup_spaces()
        obs_shape = self._get_obs_shape()
        
        # 1. 环境时间：1维
        # 2. 全局任务进度：2维 (completed_ratio, active_ratio)
        # 3. 工作站状态：5个工作站 × 3个特征 = 15维
        # 5. 全局KPI：1维 (avg_cumulative_utilization)
        global_state_dim = 1 + 2 + len(WORKSTATIONS) * 3 + 1
        self.global_state_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(global_state_dim,), dtype=np.float32)
        
        self.max_steps = self.sim.config.get("MAX_SIM_STEPS", 1500)
        self.step_count = 0
        self.render_mode = None

        self._terminal_bonus_given = False
        self._terminal_score_baseline_ema = float(REWARD_CONFIG.get('terminal_score_bonus_baseline_value', 0.0))

        # 修复缺陷二：一次性创建静态元数据
        self.obs_meta = {
            'agent_feature_names': [
                'station_id_one_hot', 'capacity_norm', 'busy_ratio', 'is_failed'
            ],
            'global_feature_names': [
                'time_progress', 'wip_rate', 'bottleneck_congestion', 'queue_len_norm'
            ],
            'queue_summary_feature_names': [
                'proc_time', 'remaining_ops', 'remaining_total_time', 'downstream_congestion', 'priority', 'is_final_op'
            ],
            'queue_summary_stat_names': [
                'min', 'max', 'mean', 'std', 'median'
            ],
            'candidate_feature_names': [
                'exists', 'remaining_ops', 'total_remaining_time', 'current_op_duration',
                'downstream_congestion', 'priority', 'is_final_op', 'product_id', 'time_pressure', 'slack'
            ],
            'normalization_constants': {
                'max_op_duration_norm': ENHANCED_OBS_CONFIG["max_op_duration_norm"],
                'max_bom_ops_norm': ENHANCED_OBS_CONFIG["max_bom_ops_norm"],
                'total_remaining_time_norm': ENHANCED_OBS_CONFIG["total_remaining_time_norm"],
                'slack_time_norm': float(REWARD_CONFIG.get('slack_tardiness_normalize_scale', 480.0)),
            },
            'num_stations': len(WORKSTATIONS),
            # 移除固定的动作空间大小，因为它现在是异构的
            # 'action_space_size': ACTION_CONFIG_ENHANCED.get('action_space_size'),
            # MultiDiscrete结构确认（供外部策略/评估校验）
            'multi_discrete_num_heads': getattr(self, '_multi_discrete_num_heads', None),
            'multi_discrete_action_dim': getattr(self, '_multi_discrete_action_dim', None),
            'multi_discrete_heads_equal_dim': True,
            'action_names': ACTION_CONFIG_ENHANCED.get('action_names'),
            'candidate_action_start': self.sim._candidate_action_start,
            'candidate_action_end': self.sim._candidate_action_end,
        }
    
    # 重写observation_space和action_space方法
    def observation_space(self, agent: str = None):
        return self._observation_spaces[agent]
    
    def action_space(self, agent: str = None):
        return self._action_spaces[agent]
        
    def _get_obs_shape(self) -> Tuple[int,]:
        # 创建一个临时的、功能齐全的仿真实例来获取状态维度
        temp_sim = WFactorySim(self.config)
        temp_sim.reset()
        # 假设所有智能体的观测空间相同
        agent_id = temp_sim.agents[0]
        obs = temp_sim.get_state_for_agent(agent_id)
        return obs.shape

    def _setup_spaces(self):
        obs_shape = self._get_obs_shape()
        self._observation_spaces = {
            agent: gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )
            for agent in self.agents
        }
        # 动作空间大小应为 1(IDLE) + 候选数量
        action_size = 1 + int(ENHANCED_OBS_CONFIG.get("num_candidate_workpieces", 0))
        
        #为每个agent定义异构的、支持并行的动作空间
        self._action_spaces = {}
        # 🔧 V2 核心修复：为支持共享网络，将动作空间填充为同构
        max_machine_count = 0
        for station_config in WORKSTATIONS.values():
            max_machine_count = max(max_machine_count, station_config.get("count", 1))

        # 保存MultiDiscrete结构确认信息，供下游组件检查/记录
        self._multi_discrete_num_heads = int(max_machine_count)
        self._multi_discrete_action_dim = int(action_size)

        for agent in self.agents:
            # 所有智能体的动作空间都填充到最大机器数，以支持共享策略网络
            self._action_spaces[agent] = gym.spaces.MultiDiscrete([action_size] * max_machine_count)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.sim.reset()
        self.step_count = 0
        self.agents = self.possible_agents[:]

        self._terminal_bonus_given = False

        observations = {agent: self.sim.get_state_for_agent(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        global_state = self.sim.get_global_state()
        for agent_id in self.agents:
            infos[agent_id]['global_state'] = global_state
            infos[agent_id]['obs_meta'] = self.obs_meta

            station_name = agent_id.replace("agent_", "")
            candidate_list = self.sim._get_candidate_workpieces(station_name)
            candidates_map = []
            for i, c in enumerate(candidate_list):
                action = self.sim._candidate_action_start + i
                candidates_map.append({
                    'action': action,
                    'queue_index': c.get('index'),
                    'part_id': c.get('part').part_id if isinstance(c.get('part'), Part) else None,
                })
            infos[agent_id]['candidates_map'] = candidates_map

            queue_snapshot = []
            for idx, part in enumerate(self.sim.queues[station_name].items):
                queue_snapshot.append({
                    'queue_index': idx,
                    'part_id': part.part_id,
                    'slack': float(calculate_slack_time(part, self.sim.env.now)),
                    'proc_time': float(part.get_processing_time()),
                })
            infos[agent_id]['queue_snapshot'] = queue_snapshot
            infos[agent_id]['action_mask'] = self.sim._get_action_mask(station_name)

        self.infos = infos
        return observations, infos

    def step(self, actions: Dict[str, int]):
        """执行一个时间步"""
        self.step_count += 1

        rewards = self.sim.step_with_actions(actions)
        observations = {agent: self.sim.get_state_for_agent(agent) for agent in self.agents}
        terminations = {agent: self.sim.is_done() for agent in self.agents}
        truncations = {agent: self.step_count >= self.max_steps for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        episode_ended = bool(any(terminations.values()) or any(truncations.values()))
        if episode_ended:
            final_stats = self.sim.get_final_stats()
            episode_score = float(calculate_episode_score(final_stats, config=self.config))
            for agent_id in self.agents:
                infos[agent_id]["final_stats"] = final_stats
                infos[agent_id]["episode_score"] = episode_score

            if (not self._terminal_bonus_given) and bool(REWARD_CONFIG.get('terminal_score_bonus_enabled', False)):
                baseline_mode = str(REWARD_CONFIG.get('terminal_score_bonus_baseline_mode', 'ema'))
                if baseline_mode == 'none':
                    baseline = 0.0
                elif baseline_mode == 'fixed':
                    baseline = float(REWARD_CONFIG.get('terminal_score_bonus_baseline_value', 0.0))
                else:
                    baseline = float(self._terminal_score_baseline_ema)

                delta = float(episode_score) - float(baseline)
                if bool(REWARD_CONFIG.get('terminal_score_bonus_positive_only', False)):
                    if delta < 0.0:
                        delta = 0.0
                clip_abs = float(REWARD_CONFIG.get('terminal_score_bonus_clip_delta_abs', 0.0))
                if clip_abs > 0.0:
                    delta = float(np.clip(delta, -clip_abs, clip_abs))

                scale = float(REWARD_CONFIG.get('terminal_score_bonus_scale', 0.0))
                bonus_total = float(scale) * float(delta)
                if bonus_total != 0.0:
                    per_agent = float(bonus_total) / float(max(1, len(rewards)))
                    for agent_id in rewards:
                        rewards[agent_id] += per_agent
                    for agent_id in infos:
                        infos[agent_id]['terminal_score_bonus'] = float(per_agent)
                        infos[agent_id]['episode_score_baseline'] = float(baseline)
                        infos[agent_id]['episode_score_delta'] = float(delta)

                if baseline_mode == 'ema':
                    alpha = float(REWARD_CONFIG.get('terminal_score_bonus_ema_alpha', 0.05))
                    alpha = float(np.clip(alpha, 0.0, 1.0))
                    self._terminal_score_baseline_ema = (1.0 - alpha) * float(self._terminal_score_baseline_ema) + alpha * float(episode_score)
                self._terminal_bonus_given = True

        global_state = self.sim.get_global_state()
        for agent_id in self.agents:
            infos[agent_id]['global_state'] = global_state
            infos[agent_id]['obs_meta'] = self.obs_meta

            station_name = agent_id.replace("agent_", "")
            candidate_list = self.sim._get_candidate_workpieces(station_name)
            candidates_map = []
            for i, c in enumerate(candidate_list):
                action = self.sim._candidate_action_start + i
                candidates_map.append({
                    'action': action,
                    'queue_index': c.get('index'),
                    'part_id': c.get('part').part_id if isinstance(c.get('part'), Part) else None,
                })
            infos[agent_id]['candidates_map'] = candidates_map

            queue_snapshot = []
            for idx, part in enumerate(self.sim.queues[station_name].items):
                queue_snapshot.append({
                    'queue_index': idx,
                    'part_id': part.part_id,
                    'slack': float(calculate_slack_time(part, self.sim.env.now)),
                    'proc_time': float(part.get_processing_time()),
                })
            infos[agent_id]['queue_snapshot'] = queue_snapshot
            infos[agent_id]['action_mask'] = self.sim._get_action_mask(station_name)

        self.infos = infos

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos
    
    def render(self, mode="human"):
        self.render_mode = mode
        if mode == "human":
            print(f"仿真时间: {self.sim.current_time:.1f}")

            print(f"完成零件数: {len(self.sim.completed_parts)}")
            for station_name in WORKSTATIONS.keys():
                queue_len = len(self.sim.queues[station_name].items)
                busy_count = self.sim.equipment_status[station_name]['busy_count']
                print(f"{station_name}: 队列={queue_len}, 忙碌设备={busy_count}")
    
    def close(self):
        """关闭环境"""
        pass

# =============================================================================
# 4. 环境工厂函数 (Environment Factory Functions)
# =============================================================================

def make_parallel_env(config: Dict[str, Any] = None):
    """直接创建PettingZoo环境（统一环境工厂入口）

    支持场景分发：
    - 默认/ scenario='factory'：W工厂生产调度环境
    - scenario='delivery'：运力商圈骑手派单调度环境（比赛场景）
    """
    # 仅在主进程中显示环境创建日志，避免worker重复输出
    import os
    try:
        import multiprocessing as _mp
        is_main_process = (_mp.current_process().name == 'MainProcess')
    except Exception:
        is_main_process = True

    cfg = dict(config or {})
    scenario = str(cfg.get('scenario', 'factory')).lower()
    if scenario == 'delivery':
        # 运力商圈配送调度（比赛场景），延迟导入避免循环依赖
        from .delivery_env import make_parallel_delivery_env
        return make_parallel_delivery_env(config)
    return WFactoryEnv(config)