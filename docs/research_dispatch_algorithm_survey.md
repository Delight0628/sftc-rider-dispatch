# 调研草稿：骑手派单 / 动态取送货调度的算法路线全网调研（2026-10）

> **状态：调研草稿（docs 约定，非 `project.md` 约束）。**
> 调研方式：全网论文检索 + GitHub 仓库逐一实际访问核实（核实日期 2026-10-08）。
> 所有论文 / 仓库 / 结论均标注来源；未能核实的仓库与论文如实标注"无法核实"，不做臆造。

---

## 1. 问题定性

### 1.1 问题特征

| 特征 | 描述 |
|---|---|
| 智能体（骑手） | 数量 **不固定、动态变化**（上线/离线），异质（位置、载具、容量、在手订单、在线意愿不同） |
| 任务（订单） | 数量 **不固定、随机到达**，带 **截止时间 / 时间窗**、取-送两段（pickup & delivery）、可合单（一对多 / 多对一） |
| 决策 | 每个决策时刻做「订单-骑手」分配（matching/dispatching），无固定 agent-task 对应关系 |
| 问题类 | **随机、动态、异构规模的组合分配问题**（stochastic task assignment / dynamic pickup-and-delivery / online bipartite matching + routing），而非静态一次性优化 |
| 优化目标 | 长期全局效率（履约率、准时率、骑手收入/效率、空驶比），而非单步贪心 |
| 关键难点 | 联合动作空间随规模变化、策略间相互干扰（非平稳性）、奖励延迟与外生性（今天的派单改变未来的供需分布）、在线毫秒级延迟约束 |

### 1.2 学术归属

该问题横跨三条文献线：
1. **在线派单 / 匹配**（order dispatching, online matching）——DiDi/Meituan/Uber 一系（见 §4c 与参考文献）；
2. **动态取送货（DPD / PDPTW / DVRP）**——运筹学动态车辆路径线（Pillac et al., 2013 综述 [R21]；Bent & Van Hentenryck, 2004 随机规划 [R20]）；
3. **多智能体强化学习（MARL）与神经组合优化（NCO）**——方法论线（见 §4a/§4b）。

三者的正确"接口"是：**中央决策的、每步求解一个（带容量/时间约束的）二分图匹配或小规模取送货插入问题**，学习的部分是"给每条 (骑手, 订单) 边打分"或"给匹配结果估值"。

---

## 2. 候选技术路线总对比表

评分说明：✅ 原生支持/强；⚠️ 部分支持/需改造；❌ 弱/无。"工业证据"指是否有 DiDi/Meituan/Amazon 等生产系统公开验证。

| 维度 | a) CTDE-MAPPO / IPPO | b) NCO（AM/POMO/MatNet/EAS/DeepACO…） | c) 学习值/边效用 + 二分图匹配（工业派单范式） | d) 平均场 MARL | e) MPC / 随机规划 / OR-Tools | f) 分层 RL / 事件驱动 |
|---|---|---|---|---|---|---|
| 变 agent 数（骑手上下线） | ⚠️ padding+mask+one-hot 到 `N_max`，非原生 | ✅ 注意力/集合编码天然吃变长集合 | ✅ 边打分函数对任意 M×N 原生 | ✅ 均值场对 N 不敏感 | ✅ 模型按当前规模建 | ✅（分层解耦规模） |
| 变 task 数（订单随机到达） | ⚠️ 候选集截断（Top-K） | ✅ 同上 | ✅ 同上 | ⚠️ | ✅ 滚动时域吸收到达 | ✅ |
| 非平稳性处理 | ❌ on-policy + 策略互扰是主要痛点 | ✅ 单一中央策略，无此问题 | ✅ 中央打分 + off-policy/保守更新 | ⚠️ 均值场平滑 | ✅ 无学习则无此问题 | ⚠️ |
| 信用分配 | ❌ 全局奖励 + GAE，弱 | ⚠️ 整解奖励（REINFORCE），粗 | ✅ 边级/反事实（COMA 式、IPS/DR） | ⚠️ 均值场近似 | ✅ 目标显式可分解 | ⚠️ 分层后改善但引入子目标信用问题 |
| 样本效率 | ❌ on-policy 低 | ⚠️ 中（REINFORCE）；离线模仿可补 | ✅ off-policy / bandit / 日志数据可复用 | ⚠️ | —（无训练） | ⚠️ |
| 在线推理延迟 | ✅ 前向一次/agent | ⚠️ 自回归 O(n) 串行解码；批量快 | ✅ 编码一次 + Hungarian/auction O(n³)（稀疏候选后近线性） | ✅ | ⚠️ 求解耗时，需分解 | ✅ 事件驱动降频 |
| 跨规模泛化 | ❌ 训练规模绑定（padding 到 `N_max`，换规模常需重训） | ✅ BQ-NCO/LEHD/EAS 报告 1000 节点泛化（RL4CO 亦强调泛化/可扩展性评测 [R30]） | ✅ 结构上天然泛化 | ✅ | ✅ | ⚠️ |
| 可行性保证（一人一单等约束） | ❌ 需后处理 | ⚠️ masking 可行但复杂约束（合单、容量）难 | ✅ 匹配层精确保证 | ❌ | ✅ 约束建模精确 | ⚠️ |
| 工业落地证据 | ❌ 无公开生产派单案例 | ❌ 无（多为 TSP/CVRP 学术基准） | ✅ **DiDi/Meituan/Amazon 均为此范式** | ⚠️ DiDi 车辆调度（Lin et al. 2018） | ✅ 业界标准兜底 | ⚠️ 学术 + 少量工业 |
| 可复现性 | ✅ on-policy/EPyMARL/pymarl2 成熟 | ✅ RL4CO 统一基准 | ⚠️ 工业系统闭源；学术混合实现（HybridMADRL-AMoD、RideGym） | ⚠️ 实现分散（XuanCe 有 MFQ/MFAC） | ✅ OR-Tools/PyVRP/VROOM | ⚠️ 多为论文自建仿真 |

**一句话结论（详见 §5）**：当前最优不是纯 MAPPO，而是 **"图/集合编码 + 学习的边效用（值函数）+ 约束二分图匹配 + 滚动时域事件驱动重优化"的混合框架**。

---

## 3. GitHub 仓库核实清单（2026-10-08 实测）

### 3.1 直接访问核实（打开仓库页面确认存在、有代码、有 README）

| 仓库 | 用途 | 核实结果 |
|---|---|---|
| [ai4co/rl4co](https://github.com/ai4co/rl4co) | NCO 统一基准（POMO/MatNet/EAS/DeepACO/PolyNet/NeuOpt/L2D/AM-PPO 等） | ✅ `pip install rl4co`；KDD 2025 论文；`rl4co/models/zoo` 含 `pomo, matnet, eas, active_search, deepaco, polynet, neuopt, l2d, amppo, symnco, gfacs, glop, dact, n2s, mdam, mvmoe, ham, nargnn, ptrnet` 等目录（main 分支最新提交 2026-09） |
| [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy) | **MAPPO 官方实现**（Yu et al., NeurIPS 2022 D&B, arXiv:2103.01955） | ✅ README/训练脚本完整；支持 SMAC/SMACv2/Hanabi/MPE/GRF；默认参数共享；另有关联 [marlbenchmark/off-policy](https://github.com/marlbenchmark/off-policy) |
| [uoe-agents/epymarl](https://github.com/uoe-agents/epymarl) | EPyMARL（Papoudakis et al., NeurIPS 2021, arXiv:2006.07869） | ✅ 含 MAPPO/IPPO/IA2C/MAA2C/MADDPG/IQL/VDN/QMIX/QTRAN/COMA/PAC；支持个体奖励、可关参数共享；Gymnasium/PettingZoo/VMAS/SMACv2 |
| [google/or-tools](https://github.com/google/or-tools) | 运筹基线：CP-SAT、VRP/TSP、linear sum assignment（匈牙利） | ✅ Apache-2.0，C++/Python/Java 封装，文档 https://developers.google.com/optimization/ |
| [ydonnelly/POMO](https://github.com/ydonnelly/POMO) | POMO 论文声称的官方实现 | ❌ **返回 404**（`ydonnelly` 现为无关个人账号，仅含 ESP8266 等仓库）。POMO 官方代码已不可用，见 §4b |
| [NExTplusplus/L2I](https://github.com/NExTplusplus/L2I) | 名为 "L2I" 的仓库 | ⚠️ 存在但 **与组合优化无关**（NLP 问答 "Learning to Imagine"，TAT-QA 竞赛代码），不能作为 NCO 的 L2I 实现 |

### 3.2 经论文/官方页面链接核实（链接来自论文正文或权威索引页，有代码与 README 描述）

| 仓库 / 数据 | 对应论文 | 核实来源 |
|---|---|---|
| [henry-yeh/DeepACO](https://github.com/henry-yeh/DeepACO) | DeepACO, NeurIPS 2023 [R8] | 论文正文给出链接；仓库页显示 ~200 stars / 30 forks / 120 commits / MIT |
| [ahottung/EAS](https://github.com/ahottung/EAS) | Efficient Active Search, ICLR 2022 [R7] | ICLR 版论文正文："Our source code is available at https://github.com/ahottung/EAS" |
| [tumBAIS/HybridMADRL-AMoD](https://github.com/tumBAIS/HybridMADRL-AMoD) | Hybrid MADRL + 加权二分图匹配, L4DC 2023 [R27] | PMLR v211 论文正文给出链接 |
| [RS2002/RideGym](https://github.com/RS2002/RideGym) | RideGym：首个 MARL 派单标准化 Gym 仿真，2026 [R28] | 论文正文给出链接，`pip install ride-gym` |
| [ricgama/maenvs4vrp](https://github.com/ricgama/maenvs4vrp) | MAEnvs4VRP 多智能体 VRP 环境库（动态/随机/多任务变体）[R29] | 论文正文给出链接 |
| [agi-brain/xuance](https://github.com/agi-brain/xuance) | XuanCe 框架（含 **MFQ/MFAC**、COMA、VDAC、MAPPO 等 40+ 算法）[R36] | arXiv:2312.16248 + PyPI 页给出链接 |
| [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl) / [hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2) / [starry-sky6688/MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms) | PyMARL（SMAC, arXiv:1902.04043）；PyMARL2（信用分配改进 QMIX）；MARL-Algorithms（含图/通信 MARL） | MARLlib 官方基准页逐一列出链接 [R37] |
| [FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL) | JaxMARL, NeurIPS 2024 D&B（GPU 加速、含变 agent 数环境）[R34] | 论文正文给出链接 |
| [facebookresearch/BenchMARL](https://github.com/facebookresearch/BenchMARL) | BenchMARL（TorchRL 后端 MARL 基准）[R35] | 论文正文给出链接 |
| [wenhaomin/LaDe](https://github.com/wenhaomin/LaDe) + [HF Cainiao-AI/LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) | **LaDe**：首个公开工业末端揽派数据集（阿里菜鸟，1067 万包裹 / 2.1 万快递员 / 6 个月 / 5 城，含取+派+轨迹+路网），KDD 2024 [R24] | 北交大官方新闻页 + 论文给出链接 |
| [aws-samples/amazon-sagemaker-amazon-routing-challenge-sol](https://github.com/aws-samples/amazon-sagemaker-amazon-routing-challenge-sol) | Amazon Last Mile Routing Research Challenge（2021，229 队参赛）AWS 参赛方案 [R25] | 论文正文给出链接 |
| [yaofengming1999/polo-courier](https://github.com/yaofengming1999/polo-courier) | POLO：多平台即时配送 MARL 派单（部分可观测 + 反事实奖励塑形），2026 [R31] | 论文正文给出链接 |
| [emiletimothy/Mean-Field-Subsample-Q-Learning](https://github.com/emiletimothy/Mean-Field-Subsample-Q-Learning) | 均值场子采样 Q 学习（2025）[R38] | 论文正文给出链接 |
| [ai4co/real-routing-nco](https://github.com/ai4co/real-routing-nco) | RRNCO：真实路网（非欧距离矩阵）NCO + 100 城真实数据集 [R39] | 论文正文给出链接 |
| [yining043/NeuOpt](https://github.com/yining043/NeuOpt) | NeuOpt：学习式 k-opt 改进搜索，NeurIPS 2023 [R40] | 论文正文给出链接 |
| [PyVRP/PyVRP](https://github.com/PyVRP/PyVRP)、[VROOM-Project/vroom](https://github.com/VROOM-Project/vroom)、[N-Wouda/ALNS](https://github.com/N-Wouda/ALNS)、[hubbs5/or-gym](https://github.com/hubbs5/or-gym) | 现代 VRP 求解器（含取送货/时间窗）与 OR-RL 环境 | GitHub `vehicle-routing-problem` topic 页列出（PyVRP 明确支持 pickup-and-delivery） |

### 3.3 无法核实 / 确认不存在（如实说明）

| 名称 | 结论 |
|---|---|
| `ydonnelly/POMO` | ❌ 404（见 §3.1）。POMO 论文 [R3] §5 声称公开 PyTorch 实现，但当前无可用官方仓库；**请用 RL4CO 的 `pomo` 实现复现** |
| "Opt-MAT" | ❌ 全网搜索无任何结果，无法确认该名称对应的真实仓库；MatNet 亦**无官方开源实现**（NeurIPS 2021 论文未提供代码链接；Samsung SDS 交流材料称 MatNet 尚处研究阶段）→ 用 RL4CO 的 `matnet` 实现 |
| `mappo-dispatch` | ❌ GitHub 仓库搜索无精确匹配（仅见无关仓库简介中出现 "MAPPO ... dispatch" 字样），无法确认存在 |
| `h-gcn`（派单语境） | ❌ GitHub 搜索 271 个结果全部为图卷积同名/衍生仓库（节点分类、骨架识别等），**无任何派单/调度相关仓库** |
| `trajkit` | ❌ GitHub 仅 6 个 0–2 star 的无关个人轨迹小工具，无派单/MARL 相关仓库 |
| BQ-NCO 官方代码 | ❌ 未发现官方仓库。ICLR 2023 投稿被拒的元评审明确指出 "the absence of code did not help"（[OpenReview 5ZLWi--i57](https://openreview.net/forum?id=5ZLWi--i57)）；截至 2026-10，`rl4co/models/zoo` 亦无 `bq_nco` 目录。其思想（利用 CO 问题对称性做 MDP bisimulation 商以提升跨规模泛化）仍值得参考 |
| "L2I"（NCO 方法） | ❌ 无法核实存在同名 NCO 方法/仓库（同名仓库为 NLP 代码）。"Learning-to-Improve" 实为**改进式（L2S）范式**的泛称；可核实的同类工作为 NeuOpt、DACT、Neural LNS 等（RL4CO zoo 内含） |

---

## 4. 各技术路线详细评估

### 4a. CTDE-MAPPO / IPPO（on-policy MARL）

**代表工作与实现**
- MAPPO：Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 D&B（arXiv:2103.01955 [R4]）；官方代码 [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy)（已核实）。
- IPPO / MADDPG / IA2C / QMIX 系：[uoe-agents/epymarl](https://github.com/uoe-agents/epymarl)（已核实，支持个体奖励、可关参数共享）；[hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)（QMIX+信用分配改进）。
- 变长 agent 的常见做法：参数共享 + agent one-hot/属性编码 + `N_max` padding + mask（项目现状即此路线）。

**优点**：CTDE 生态成熟、代码稳定、小中规模下训练可靠；集中式 critic 能缓解部分非平稳性；参数共享对同质骑手合理。

**缺点（对该问题类别）**：
1. **变规模非原生**：动作/观测维随 `N_max` 固定；骑手上下线靠 padding/mask，跨规模泛化差，重训成本高；
2. **非平稳性**：联合策略同时更新 + 策略互扰，on-policy 又不允许大量复用旧数据，样本效率低（on-policy README 自己提醒"很多论文没复现对"，调参敏感）；
3. **信用分配弱**：全局奖励 + GAE 无法区分"哪个骑手接哪单"的边际贡献（对比 COMA 式反事实或边级奖励）；
4. **可行性不保证**：各骑手独立 argmax 会产生冲突（多骑手抢单/订单无人接），最终仍要后处理成匹配问题——即 MAPPO 的输出其实只是匹配层的打分；
5. **无工业派单证据**：公开的 DiDi/Meituan/Amazon 生产派单系统均不是纯 MAPPO 范式。

**定位**：保留为研究基线（消融对照），或仅用于"上层策略"（如是否等待、是否合单的离散决策），不建议作为派单主干。

### 4b. GNN + 注意力的神经组合优化（NCO）

**代表工作**：Attention Model（Kool et al., ICLR 2019 [R2]）、POMO（Kwon et al., NeurIPS 2020 [R3]）、MatNet（Kwon et al., NeurIPS 2021，**矩阵/二组元输入**，天然对应"骑手×订单关系矩阵" [R5]）、BQ-NCO（Drakulic et al.，bisimulation 商提升跨规模泛化至 1000 节点 [R6]）、EAS（Hottung et al., ICLR 2022，测试时只更新部分参数的实例级搜索，CVRP 上超越 LKH3 [R7]）、DeepACO（Ye et al., NeurIPS 2023，神经增强蚁群 [R8]）、GLOP/PolyNet/NeuOpt/NDS（大规模与改进式代表 [R40][R41]）。

**实现**：[ai4co/rl4co](https://github.com/ai4co/rl4co)（已核实）是当前最可信的统一复现入口（POMO/MatNet/EAS/DeepACO/PolyNet/NeuOpt/L2D/AM-PPO 均在 `models/zoo`）；EAS 另有官方 [ahottung/EAS](https://github.com/ahottung/EAS)；DeepACO 另有官方 [henry-yeh/DeepACO](https://github.com/henry-yeh/DeepACO)。POMO/MatNet 官方实现缺失（§3.3）。

**优点**：
1. **变规模原生**：注意力/DeepSets 编码器对任意大小的骑手集合与订单集合直接工作，无需 padding；
2. **跨规模泛化有实证**：BQ-NCO、LEHD、EAS 均报告小规模训练 → 1000 节点推理；
3. GPU 批量推理快，RL4CO 提供 TorchRL/TensorDict 级工程效率；
4. MatNet 的"双组元矩阵编码"与派单的二部结构完全同构。

**缺点**：
1. **建模错位**：NCO 面向"静态单实例、终局奖励、构造式解"；派单是"随机到达、延迟回报、多轮滚动决策"。把派单塞进 NCO 需自定义环境/奖励（RL4CO 支持自定义 env，但非开箱即用）；
2. **自回归解码延迟**：一次完整构造 O(n) 串行 softmax，大规模匹配（数千×数千）不如一次矩阵打分 + 匹配；
3. 训练多为 REINFORCE，样本效率中等；对在线安全探索没有机制；
4. 无多智能体概念——但这其实是优点：中央构造式策略规避了 MARL 非平稳性，代价是动作规模被串行化。

**定位**：**编码器与打分头的首选来源**（尤其 MatNet/POMO 的 encoder + RL4CO 工程栈），以及"单骑手路径/取送顺序"子问题的策略来源；不建议直接把整场派单当一个巨型 TSP 解。

### 4c. 学习值/边效用 + 二分图匹配（工业派单范式）★工业验证最强

**工业证据链（全部为公开论文）**：
- **DiDi**：Xu et al., KDD 2018（"学习+规划"：时空分位量化学长期价值 + 实时组合优化匹配，**已上线生产**）[R9]；Tang et al., KDD 2019（**deep value network 估值 + 多司机派单**，大规模线上 A/B）[R10]；Qin et al., INFORMS J. Applied Analytics 2020（DiDi 派单 RL 全流程工程报告）[R11]；Sadeghi Eshkevari et al., KDD 2022（"RL in the Wild"：TD 值更新 + **maximum bipartite matching** + MAB 图剪枝，多城市 A/B +1.3% 司机收入、全量后因果推断 +5.3%，并成为某国际主力市场默认派单模式）[R12]；
- **Meituan**：SCDN（图表示学习挖掘熟练骑手合单模式 → 向量近邻剪枝实时 many-to-one 分配，**已部署**，午高峰骑手效率 +45–55%）[R13]；MRGRP（多关系图 GraphFormer 骑手路径预测，**已部署**美团 Turing 平台）[R14]；Auad et al., Transportation Science 2026（n-step SARSA 值函数 + MAB 超启发式 7 个低层启发式，美团真实数据 12% 降本）[R15]；
- **Amazon**：Last Mile Routing Research Challenge（2021，229 队）[R25]；AWS 方案 = 学习司机顺序概率模型 + Rollout + 经典 TSP 求解器分层结合 [R26]；
- **学术侧同构**：Enders et al., L4DC 2023（多智能体 SAC 估值 + **加权二分图匹配**分解联合动作空间，真实出租车数据超越 SOTA）[R27]；RideGym（2026，首个标准化 MARL 派单 Gym，实验也表明探索噪声会改变 MARL 排名）[R28]。

**优点**：
1. 变规模原生（边打分 f(骑手特征, 订单特征, 上下文) 对任意 M×N）；
2. **匹配层精确保证可行性**（一人一单/一单一人/容量 b-matching），且 Hungarian/auction 复杂度可控；
3. 信用分配可下沉到边级（反事实优势、IPS/DR 离线评估），样本可用历史日志（off-policy / contextual combinatorial bandit）；
4. 延迟可控、可解释（效用函数可分解为 ETA、时间压力、骑手收入等项——DiDi KDD 2022 明确强调可解释定制效用函数）；
5. **是唯一被多家生产系统公开验证的范式**。

**缺点**：
1. 打分函数若只用线性/浅层会 myopic——需要用图/注意力编码器升级（即与路线 b 融合）；
2. 匹配层与值估计联合训练困难（目标不可微），常用"匹配求解 + 反事实评估"交替；
3. 合单/多单骑手需要升级为带容量匹配或与路线 f 的路径子问题联动；
4. 工业系统闭源，学术复现要靠 HybridMADRL-AMoD / RideGym / 自建仿真。

### 4d. 平均场 MARL / 随机图策略（large-scale MARL）

**代表工作**：Yang et al., ICML 2018（MFQ/MFAC，成对交互均值场近似 [R16]）；DiDi Lin et al., KDD 2018（mean field 式大规模车队管理，600+ 引用 [R17]）；近代理论改进：非均匀交互下 MFC 近似误差界（UAI 2022 [R18]）、子采样均值场 Q（2025 [R38]）。**实现**：XuanCe 的 MFQ/MFAC（[agi-brain/xuance](https://github.com/agi-brain/xuance)，已核实文档）、[emiletimothy/Mean-Field-Subsample-Q-Learning](https://github.com/emiletimothy/Mean-Field-Subsample-Q-Learning)。

**优点**：对数千 agent 可扩展（O(N) 而非 O(N²)），适合"群体密度/供需均衡"类决策（如再平衡 repositioning）。

**缺点（对该问题类别）**：
1. **同质匿名交互假设不成立**：骑手异质（容量、在手单、速度、区域），订单与骑手是二部匹配关系而非群体碰撞；
2. 均值场只保留"平均邻居动作"，丢失"这单给谁最优"的排他性约束——恰恰派单的核心；
3. 工业证据集中在车队调度/再平衡而非逐单匹配；非均匀交互下近似误差有理论缺口 [R18]；
4. 实现分散，无统一权威基准。

**定位**：仅当问题扩展到"骑手再平衡/区域运力调控"子问题且规模极大时考虑；不作为逐单派单主干。

### 4e. 模型预测控制 / 随机规划 / 运筹优化基线

**代表工作**：Alonso-Mora et al., PNAS 2017（滚动窗口 + MIP 的实时高容量拼车匹配，Uber/MIT 实验数据 [R19]）；Bent & Van Hentenryck 2004（场景近似 SAA + 两阶段局部搜索求解随机 VRP [R20]）；Pillac et al., EJOR 2013（动态 VRP 综述，rolling horizon/在线重优化分类 [R21]）。**工具**：[google/or-tools](https://github.com/google/or-tools)（CP-SAT、VRP、linear sum assignment，已核实）、[PyVRP](https://github.com/PyVRP/PyVRP)（支持取送货/时间窗/多行程）、[VROOM](https://github.com/VROOM-Project/vroom)（PDPTW）、[ALNS](https://github.com/N-Wouda/ALNS)、[or-gym](https://github.com/hubbs5/or-gym)。

**优点**：可行-by-构造、可解释、约束表达完整、零训练、强基线；滚动时域天然吸收订单随机到达与骑手上下线；SAA 处理需求不确定性有成熟理论。

**缺点**：依赖预测分布质量；大规模求解延迟（需候选剪枝/分解）；对"长期价值/ETA 模型/用户行为"这类黑盒目标表达力弱；不能从数据自动改进。

**定位**：**基线 + 兜底 + 混合框架的"匹配/重优化层"**。任何学习方法都必须先打赢"滚动时域 + Hungarian/CP-SAT"这条线。

### 4f. 分层强化学习 / 事件驱动 dispatching（DEDS 建模）

**代表工作**：Shiri et al., IEEE T-ITS 2025（HRL actor-critic + H3 空间分区，联合优化 matching 与 dispatching [R22]）；Auad et al. 2026（上层 n-step SARSA + 下层 MAB 超启发式，本质即分层 [R15]）；POLO 2026（平台-网格 agent 分解 + 注意力聚合 + 反事实奖励塑形应对联合动作非平稳 [R31]）。事件驱动/DEDS 建模将"何时决策"（订单到达、骑手上/下线、完成服务）作为事件流，降低决策频率（对比固定 tick）。

**优点**：把"是否等待/合单、派给谁、路径顺序、再平衡"分层解耦，动作空间爆炸问题缓解；事件驱动贴合真实系统（DiDi KDD 2022 即按派单周期 + 反馈控制机制运行）。

**缺点**：分层训练复杂、子目标/时间信用分配难、收敛慢；开源实现多为论文自建仿真（无权威库）；HRL 超参与层级设计对效果敏感。

**定位**：当问题扩展为"派单 + 合单 + 路径 + 再平衡"联合优化时的**组织框架**，其下层仍应是"学习打分 + 匹配"。

---

## 5. 客观最优推荐（截至 2026）

### 5.1 推荐框架：学习打分 + 约束匹配 + 滚动重优化的混合架构

```
事件驱动（订单到达 / 骑手上·下线 / 服务完成）
   │
   ▼
[图/集合编码器]  骑手集合(变长) + 订单集合(变长) + 二部关系矩阵
   │   （MatNet/AM/POMO 式注意力编码，RL4CO 工程栈；天然变规模）
   ▼
[学习的边效用/值]  q(骑手 i, 订单 j | 全局上下文)
   │   训练：off-policy TD / contextual combinatorial bandit；
   │   信用分配：边级反事实（COMA 式）或匹配对边际贡献；可用历史日志 + IPS/DR
   ▼
[约束匹配层]  加权二分图匹配 / 带容量 b-matching（Hungarian / auction / 贪心+交换）
   │   精确保证可行性；合单升级为带容量匹配或小规模插入启发式
   ▼
[滚动时域重优化]  未履约订单与空闲骑手滚动重入；兜底 OR-Tools/PyVRP 或局部搜索
```

**为什么是它（按评估维度）**：
- **变规模**：集合/图编码 + 边打分，M、N 任意变化零改造（MAPPO 需要 padding 重训）；
- **非平稳性**：中央打分模型只有一份策略，不存在 MARL 策略互扰；非平稳只来自环境（用保守/off-policy 更新应对）；
- **信用分配**：边级效用天然把全局目标分解到 (i,j) 对，配反事实基线即可；
- **样本效率**：off-policy + 日志数据（DiDi/Meituan 路线的共同选择），远高于 on-policy PPO；
- **延迟**：一次编码 + O(n³) 匹配；候选剪枝（Meituan SCDN 的向量近邻剪枝 [R13]、DiDi 的 MAB 图剪枝 [R12]）后可近线性；
- **跨规模泛化**：NCO 侧有 BQ-NCO/LEHD/EAS 的千节点实证 [R6][R7]，匹配层结构上无关规模；
- **工业证据**：DiDi KDD 2018/2019/2022、Meituan SCDN/MRGRP、Amazon Last Mile 全部是"学习 + 匹配/运筹"混合 [R9][R10][R12][R13][R25]；
- **可复现性**：编码器/训练侧用 RL4CO 或 on-policy 组件，匹配侧用 OR-Tools（`linear_sum_assignment`/CP-SAT），仿真侧用 RideGym/MAEnvs4VRP/LaDe 数据 [R28][R29][R24]。

**为什么不选纯 MAPPO**：见 §4a——变规模非原生、on-policy 低效、信用分配弱、可行性不保证、跨规模差、无生产派单证据；MAPPO 的独立动作输出最终仍要过匹配层，等于用高成本学了一个"不如边打分直接"的打分器。MAPPO 适合做消融基线或上层离散决策（等待/合单开关）。

### 5.2 失败模式与预案

| 失败模式 | 表现 | 预案 |
|---|---|---|
| 训练-推理分布偏移 | 线上策略改变订单/骑手行为分布，指标回退 | 保守策略更新（trust-region/clip）、off-policy 修正、线上小流量灰度 + 因果推断评估（DiDi KDD 2022 做法） |
| 奖励延迟与外生性 | 派单效果数小时后才显现，方差大 | n-step TD（Meituan [R15]）+ 反事实基线 + 滚动回测 |
| 贪心打分短视 | 全局效率不升 | 打分含长期价值项（时空价值面，DiDi KDD 2018 [R9]）+ 匹配层全局化 |
| 匹配与路径/合单耦合 | 合单质量差 | 带容量匹配 + 小规模插入/局部搜索；参考 SCDN 合单剪枝 [R13] |
| 极端规模延迟 | 高峰 M×N 过大 | 候选剪枝（区域召回/向量近邻/MAB 剪枝）+ auction/贪心降级 |
| 仿真-现实差距 | 仿真涨、线上不涨 | 用真实日志校准仿真（接受率、服务时间模型）；参考 LaDe 数据校验 [R24]、RideGym 确定性仿真 [R28] |
| 学习方法打不过启发式 | （WareMax 基准即如实报告 RL 不如 nearest/auction [R42]） | 先打赢"滚动时域 + Hungarian"基线再上学习；学习组件只替代打分，不替代匹配 |

### 5.3 实施优先级建议（与项目现状衔接，仅建议不动代码）

1. **P0**：把匹配层显式化（哪怕先用贪心/匈牙利），动作定义从"每骑手独立选单"改为"边打分 + 匹配"；
2. **P1**：观测/编码改集合/图编码（项目已有 `plan_scheme_b_set_encoding.md` 方向一致），对齐 MatNet/AM 结构；
3. **P2**：训练信号改为边级效用 + 反事实/离线评估，摆脱 on-policy 依赖；
4. **P3**：引入滚动时域与事件驱动；以 OR-Tools/PyVRP 为兜底与对拍；
5. **P4**：若扩展合单/再平衡，引入分层组织（路线 f）。

### 5.4 落地实现（2026-10-08，P0 已完成）

P0「边打分 + 约束匹配 + 滚动重优化」已实现（P1–P4 未动）：

- `environments/hybrid_dispatch.py`：`EdgeScorer`（12 维边特征线性效用，可学习，
  weights 先验=紧迫分量+配对质量分量分解）+ `max_weight_matching`（贪心+交换改进
  二分图匹配，零依赖确定性）+ `HybridDispatcher`（每决策 epoch 重建边集，
  按容量分轮求解 = 带容量 b-matching，未派单滚动重入；`learn=True` 时按实现效用
  delta 规则回归）；
- `environments/delivery_config.py` §12：`HYBRID_DISPATCH_CONFIG`（权重/学习率/轮数上限）；
- `evaluation_delivery.py`：新增 `--baseline hybrid`（联合决策分支），与
  nearest/EDD/fifo/random 同口径对拍；
- `checks/hybrid_dispatch_check.py`：28 项校验（匹配一对一/无抢单/无非法动作/
  赢启发式/变规模 N=3/5/8/学习回路）。

**实测结论（mock 固定订单 + deterministic_candidates，seed 0/42 各 5 episodes；
真实果洛 orders_guoluo_20260915_ready + riders_full，3 episodes，均为 5 骑手）**：

| 数据 | baseline | completion | on_time | tardiness | score |
|---|---|---|---|---|---|
| mock seed=0 | edd（最强启发式） | 1.000 | 0.246 | 654.1 | 0.690 |
| mock seed=0 | **hybrid** | 1.000 | **0.859** | **32.2** | **0.925** |
| mock seed=42 | edd | 1.000 | 0.280 | 547.8 | 0.697 |
| mock seed=42 | **hybrid** | 1.000 | **0.901** | **11.3** | **0.929** |
| 真实果洛 | edd | 0.717 | 0.326 | 10176.0 | 0.386 |
| 真实果洛 | **hybrid** | **1.000** | **0.761** | **1921.0** | **0.660** |

hybrid 全程 race_conflict=0、invalid_action=0。结论：**学习打分+约束匹配混合框架
在 mock 与真实样本上均显著赢全部启发式基线**（RL 有效性铁律满足，P0 阶段
"学习"部分为手调先验+可选在线回归，尚未引入离线训练）。

历史遗留问题已修复（2026-10-08）：`checks/dispatch_integration_check.py` 槽位断言
已同步新口径（候选[1]=due_rel，意愿分仅在 `candidates_map[i]['willingness']`），
README 两处旧描述同步更正；现 14/14 通过。

**P1/P2 落地（2026-10-08，训练链路完成）**：

- `hybrid/edge_value_net.py`：`EdgeValueNet`（TF/Keras，DeepSets+多头注意力集合编码，
  N 可变）+ `NpEdgeValueNet`（npz numpy 前向，worker/评估机零 TF）；warm-start 用
  ±x 双通道使零训练时 q≡线性先验（parity 校验通过，max|Δ|<1e-3）。
- `hybrid/scorers.py` / `replay_buffer.py` / `collect.py` / `evaluate.py` /
  `trainer.py` + `hybrid_train.py` CLI：off-policy n-step TD（n=5, γ=0.99, τ=0.005
  软更新，max_grad_norm=5.0），温度采样行为策略（0.5→0.1/200iter），
  ProcessPoolExecutor 并行采集（失败降级串行），neural 连续 2 轮不赢 linear →
  回滚 best + lr 减半；产物 `checkpoints/edge_net_{best,last,final}.npz/.weights.h5`
  + `metrics.jsonl` + TensorBoard。约定详见 `docs/hybrid_implementation.md`。
- 校验：`checks/check_hybrid_training.py` 30/30（obs 结构/采集结构/回放/评估口径/
  warm-start parity/npz 双实现一致/TD 更新有限）；`set_encoding_check.py` 27/27；
  `real_pool_train_check.py` 22/22（静态断言已改 hybrid 入口）；TF 降级路径 25/25。
- mock 联调训练冒烟（2 iters，episodes_per_iter=4，seed=0）：neural 0.9097 ≥
  linear 0.9095 ≫ edd/nearest/fifo 0.666；采集-TD-评估-回滚-存档链路全通。
- 真实果洛对拍（3 episodes，neural=mock 冒烟产物）：0.473 ≈ linear 0.477
  （训练量不足，未偏离先验，符合 warm-start「不劣于先验」设计）；真实业务结论
  需在远程用果洛数据完整训练后给出。

关键 bug 修复记录：① warm-start ±x 双通道（relu 截断负值）；② 目标网络硬拷贝
（两实例前层独立随机初始化会导致 bootstrap 目标错误）；③ npz 键名 TF2.15 前缀；
④ 真实 order_id 超 int32 → cand_order_ids 改 int64。

---

## 6. 与固定工位流水车间（JSP/FJSP/DFJSP）的本质差异

| 维度 | 固定工位流水车间（JSP/FJSP） | 骑手派单 / 动态取送货 |
|---|---|---|
| 资源（agent） | 固定 m 台机器/工位 | 骑手数 **N 动态**（上下线）、异质 |
| 任务 | 工件按**固定工艺路线**过站 | 订单取-送两段、**无固定服务者**、可合单 |
| 规模 | 训练/推理规模固定 | **M、N 双向变长**，需跨规模泛化 |
| 任务到达 | 批量已知（或静态释放） | **随机在线到达** + 截止时间/时间窗 |
| 资源-任务关系 | 工序-机器兼容矩阵固定 | 二部图随上下文（时空、载量、意愿）变化 |
| 决策结构 | 排序/机器指派（每机一个队列） | **每时刻的全局匹配**（排他性约束） |
| 随机性 | 加工时间扰动为主 | 到达过程、服务时间、骑手接受行为全随机 |
| 目标 | makespan/tardiness（离线） | 长期履约/效率/收入（在线、滚动） |
| 非平稳性 | 低（环境静态） | 高（策略互改供需） |
| 算法含义 | 工序级调度策略即可 | 必须有**可行性匹配层 + 滚动重优化** |

结论：用固定 `N_max` 的 CTDE-MAPPO 直接迁移（本项目初版路线）在"规模、随机到达、排他匹配、跨规模"四点上都与问题结构错位；正确迁移对象是"边打分 + 匹配 + 滚动重优化"的派单范式（DiDi/Meituan 已验证），NCO 提供编码器，MARL 只作为可选上层。

---

## 7. 参考文献（含 URL）

**MARL / 基础算法**
- [R1] Rashid et al., "QMIX: Monotonic Value Function Factorisation", ICML 2018. https://arxiv.org/abs/1803.11485
- [R4] Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu, "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 D&B. https://arxiv.org/abs/2103.01955 （代码 https://github.com/marlbenchmark/on-policy）
- [R16] Yang, Luo, Li, Zhou, Zhang, Wang, "Mean Field Multi-Agent Reinforcement Learning", ICML 2018. https://proceedings.mlr.press/v80/yang18d.html
- [R18] Mondal, Aggarwal, Ukkusuri, "Can Mean Field Control (MFC) Approximate Cooperative MARL with Non-Uniform Interaction?", UAI 2022. https://arxiv.org/abs/2203.00035
- [R34] Rutherford et al., "JaxMARL: Multi-Agent RL Environments and Algorithms in JAX", NeurIPS 2024 D&B. https://arxiv.org/abs/2311.10090 （代码 https://github.com/FLAIROx/JaxMARL）
- [R35] Bettini, Prorok, Moens, "BenchMARL: Benchmarking Multi-Agent Reinforcement Learning". https://github.com/facebookresearch/BenchMARL
- [R36] Liu et al., "XuanCe: A Comprehensive and Unified Deep RL Library". https://arxiv.org/abs/2312.16248 （代码 https://github.com/agi-brain/xuance，含 MFQ/MFAC）
- [R37] MARLlib 基准页（PyMARL/PyMARL2/EPyMARL/MARL-Algorithms 仓库索引）. https://marllib.readthedocs.io/en/latest/_sources/resources/benchmarks.rst.txt
- [R38] Anand, Karmarkar, Qu, "Mean-Field Sampling for Cooperative MARL", 2025. https://arxiv.org/abs/2412.00661
- Papoudakis et al., "Benchmarking Multi-Agent Deep RL Algorithms in Cooperative Tasks", NeurIPS 2021 D&B. https://arxiv.org/abs/2006.07869 （EPyMARL: https://github.com/uoe-agents/epymarl）

**NCO**
- [R2] Kool, van Hoof, Welling, "Attention, Learn to Solve Routing Problems!", ICLR 2019. https://arxiv.org/abs/1803.08475
- [R3] Kwon et al., "POMO: Policy Optimization with Multiple Optima for RL", NeurIPS 2020. https://arxiv.org/abs/2010.16011 （官方代码链接已失效，见 §3.3）
- [R5] Kwon et al., "Matrix Encoding Networks for Neural Combinatorial Optimization" (MatNet), NeurIPS 2021. https://arxiv.org/abs/2106.11113
- [R6] Drakulic, Michel, Mai, Sors, Andreoli, "BQ-NCO: Bisimulation Quotienting for Generalizable Neural Combinatorial Optimization". OpenReview（ICLR 2023 投稿，元评审指出"absence of code"）: https://openreview.net/forum?id=5ZLWi--i57 ，无官方代码
- [R7] Hottung, Kwon, Tierney, "Efficient Active Search for Combinatorial Optimization Problems", ICLR 2022. https://arxiv.org/abs/2106.05126 （代码 https://github.com/ahottung/EAS）
- [R8] Ye et al., "DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization", NeurIPS 2023. https://arxiv.org/abs/2309.14032 （代码 https://github.com/henry-yeh/DeepACO）
- [R30] Berto et al., "RL4CO: an Extensive RL for CO Benchmark", KDD 2025. https://arxiv.org/abs/2306.17100 （代码 https://github.com/ai4co/rl4co）
- [R39] Son et al., "Neural Combinatorial Optimization for Real-World Routing" (RRNCO，真实路网距离/时长矩阵 + 100 城数据集). https://arxiv.org/abs/2503.16159 （代码 https://github.com/ai4co/real-routing-nco）
- [R40] Ma, Cao, Chee, "Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt" (NeuOpt), NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/file/9bae70d354793a95fa18751888cea07d-Paper-Conference.pdf （代码 https://github.com/yining043/NeuOpt）
- [R41] Hottung, Mahajan, Tierney, "PolyNet: Learning Diverse Solution Strategies for NCO". https://arxiv.org/abs/2402.14048 ；Luo et al., "LEHD: NCO with Heavy Decoder" NeurIPS 2023. https://arxiv.org/abs/2310.07985

**工业派单 / 匹配**
- [R9] Xu et al. (DiDi), "Large-Scale Order Dispatch in On-Demand Ride-Hailing Platforms: A Learning and Planning Approach", KDD 2018. https://dl.acm.org/doi/10.1145/3219819.3219824
- [R10] Tang et al. (DiDi), "A Deep Value-Network Based Approach for Multi-Driver Order Dispatching", KDD 2019.
- [R11] Qin, Tang, Jiao, Zhang, Xu, Zhu, Ye (DiDi), "Ride-Hailing Order Dispatching at DiDi via Reinforcement Learning", INFORMS J. Applied Analytics 50(5), 2020.
- [R12] Sadeghi Eshkevari, Tang, Qin, Mei, Zhang, Meng, Xu (DiDi), "Reinforcement Learning in the Wild: Scalable RL Dispatching Algorithm Deployed in Ridehailing Marketplace", KDD 2022. https://arxiv.org/abs/2202.05118
- [R13] Liang et al. (Meituan), "Harvesting Efficient On-Demand Order Pooling from Skilled Couriers" (SCDN), 2024. https://arxiv.org/abs/2406.14635
- [R14] Liu et al. (Meituan/清华), "MRGRP: Empowering Courier Route Prediction ... with Multi-Relational Graph", WWW Companion 2025. https://arxiv.org/abs/2505.11999
- [R15] Auad, Lagos, Lagos, "Data-Driven Optimization for Meal Delivery: A RL Approach for Order-Courier Assignment and Routing at Meituan", Transportation Science, 2026. https://doi.org/10.1287/trsc.2025.0129
- [R17] Lin, Zhao, Xu, Zhou, "Efficient Large-Scale Fleet Management via Multi-Agent DRL", KDD 2018（mean field 车队调度）.
- [R25] Amazon-MIT, "Amazon Last Mile Routing Research Challenge" (2021). https://www.amazon.science/last-mile-routing-research-challenge
- [R26] Wu, Song, March, Duthie (AWS), "Learning from Drivers to Tackle the Amazon Last Mile Routing Research Challenge". https://arxiv.org/abs/2205.04001
- [R27] Enders, Harrison, Pavone, Schiffer, "Hybrid Multi-agent DRL for AMoD Systems", L4DC 2023. https://arxiv.org/abs/2212.07313 （代码 https://github.com/tumBAIS/HybridMADRL-AMoD）
- [R31] Yao, Luo, "Partially Observable Learning for Multi-Platform Dispatch Optimization" (POLO), 2026. https://arxiv.org/abs/2608.10897 （代码 https://github.com/yaofengming1999/polo-courier）
- Chouakia, Hörl, Puchinger, "A review on RL methods for mobility on demand systems", 2025. https://arxiv.org/abs/2501.02569

**运筹 / 动态 VRP / 数据集**
- [R19] Alonso-Mora, Samaranayake, Wallar, Frazzoli, Rus, "On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment", PNAS 2017.
- [R20] Bent, Van Hentenryck, "Scenario-Based Stochastic Combinatorial Optimization", Operations Research 52(6), 2004；及 "A Two-Stage Hybrid Local Search for the Vehicle Routing Problem with Stochastic Demands", Transportation Science 38(4), 2004.
- [R21] Pillac, Gendreau, Guéret, Medaglia, "A review of dynamic vehicle routing problems", EJOR 225(1), 2013.
- [R22] Shiri, YarAhmadi, Keivanpour, "Real-Time Matching and Dispatching for Urban Freight ... Hierarchical RL ... H3 Spatial Partitioning", IEEE T-ITS, 2025. DOI 10.1109/TITS.2025.3601536
- [R24] Wu, Wen, Hu, Mao et al., "LaDe: The First Comprehensive Last-mile Express Dataset from Industry"（阿里菜鸟）, KDD 2024. https://arxiv.org/abs/2306.10675 （数据 https://huggingface.co/datasets/Cainiao-AI/LaDe ，代码 https://github.com/wenhaomin/LaDe）
- [R28] Zhao, Hu, Li, "RideGym: A Standardized Interface for Real-World Large-Scale Ride-Sharing System", 2026. https://arxiv.org/abs/2607.10173 （代码 https://github.com/RS2002/RideGym）
- [R29] Gama et al., "Multi-Agent Environments for Vehicle Routing Problems" (MAEnvs4VRP). https://arxiv.org/abs/2411.14411 （代码 https://github.com/ricgama/maenvs4vrp）
- [R42] Skelf Research, "WareMax：RMFS 任务分配的确定性离散事件仿真与 RL 基准"（如实报告 RL 派单未超越 nearest/auction 启发式）. https://waremax.skelfresearch.com/faq/
- OR-Tools: https://developers.google.com/optimization/ ；PyVRP: https://github.com/PyVRP/PyVRP ；VROOM: https://github.com/VROOM-Project/vroom

**补充（取送货路径预测，Meituan/北交大线）**
- Wen et al., "Graph2route: A Dynamic Spatial-Temporal GNN for Pick-up and Delivery Route Prediction", KDD 2022；Mao et al., "DRL4Route: A Deep RL Framework for Pick-up and Delivery Route Prediction", KDD 2023；Wen et al., "A Survey on Service Route and Time Prediction in Instant Delivery", TKDE 2024（综述）。

---

## 8. 附：本次调研的核实方法说明

- "已核实"= 2026-10-08 直接打开仓库页面确认 README/代码结构/许可证；或打开论文正文确认作者自带代码链接与仓库描述。
- 失效/不存在项（`ydonnelly/POMO`、`Opt-MAT`、`mappo-dispatch`、`h-gcn` 派单语境、`trajkit` 派单语境、BQ-NCO 官方代码、NCO 语境的 `L2I`）均经 GitHub 检索/直接访问确认，未在文中臆造替代链接。
- 未直接访问但有论文/权威索引背书的仓库已单独标注核实来源，使用前建议再跑一次 `git clone` 验证。
