import numpy as np
from typing import List, Sequence, Optional, Union


def _as_head_probs_list(model_out: Union[Sequence[np.ndarray], np.ndarray]) -> List[np.ndarray]:
    """将模型输出标准化为每头一个一维概率数组的列表。"""
    head_list: List[np.ndarray]
    if isinstance(model_out, (list, tuple)):
        head_list = [np.asarray(h).squeeze() for h in model_out]
    else:
        arr = np.asarray(model_out)
        if arr.ndim == 2:
            arr = arr[0]
        head_list = [arr.squeeze()]
    return [np.clip(h.astype(np.float64), 1e-12, np.inf) for h in head_list]


def choose_parallel_actions_multihead(
    head_probs: Union[Sequence[np.ndarray], np.ndarray],
    num_heads: int,
    greedy: bool = True,
    sample_eps: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    多头无放回动作选择（允许多个头选择IDLE=0）。

    参数：
    - head_probs: 模型输出（list[ndarray] 或 ndarray），每头的动作概率分布
    - num_heads: 需要输出的头数（设备数）
    - greedy: True时每头取argmax；False时允许采样
    - sample_eps: 当greedy=False时，以该概率走随机采样，否则仍取greedy（一次性决定整次调用）
    - rng: 可选随机数发生器
    返回：np.ndarray[int]，长度为num_heads
    """
    if rng is None:
        rng = np.random.RandomState()

    probs_list = _as_head_probs_list(head_probs)
    # 方案4.1：Actor可能额外输出 mixture_weights（例如长度为2的向量），不属于动作头
    # 若外部直接把整个输出列表传入，这里自动剥离末尾，避免把mixture当成一个头。
    if isinstance(probs_list, list) and len(probs_list) == (int(num_heads) + 1):
        try:
            tail = np.asarray(probs_list[-1]).squeeze()
            if tail.shape == (2,):
                probs_list = probs_list[:-1]
        except Exception:
            pass
    used_non_idle = set()  # 记录非零动作，避免重复

    # 决定本次是否整体随机采样
    do_sample = (not greedy) and (rng.rand() < max(0.0, min(1.0, sample_eps)))

    chosen: List[int] = []
    for i in range(num_heads):
        # 取对应头的分布，不足则复用第一个头
        if i < len(probs_list):
            p = probs_list[i].copy()
        else:
            p = probs_list[0].copy()

        # 掩码：无放回（允许多个头选择 IDLE(0)）
        if used_non_idle:
            idxs = [u for u in used_non_idle if u != 0 and u < p.shape[0]]
            if idxs:
                p[idxs] = 0.0

        s = p.sum()
        if not np.isfinite(s) or s <= 1e-12:
            idx = 0
        else:
            p = p / s
            if do_sample:
                idx = int(rng.choice(np.arange(len(p)), p=p))
            else:
                idx = int(np.argmax(p))

        chosen.append(idx)
        if idx != 0:
            used_non_idle.add(idx)

    return np.asarray(chosen, dtype=np.int32)


