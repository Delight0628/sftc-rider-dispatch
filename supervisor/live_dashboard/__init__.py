"""supervisor.live_dashboard — Live 调度大屏（比赛/路演最终展示形态）

架构（project.md §7.5 定稿）：
    DeliveryEnv 仿真 -> 帧序列（sim_stream）-> HTTP API（server）-> 大屏前端（static/index.html）

输出语义 = 每骑手实时订单 list（紧迫分 x 双塔偏好分 外部加权融合）。
双塔（王新春侧）数据契约：upstream_candidates = {骑手: [{order_id, willingness}]}
—— mock 替身 generate_mock_upstream_candidates 与真实上游契约完全一致，可直接替换。
"""
