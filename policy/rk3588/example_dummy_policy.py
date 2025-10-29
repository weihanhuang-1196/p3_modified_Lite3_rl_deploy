"""示例策略与环境适配代码，用于快速验证部署流程。"""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    """随机策略示例，仅用于采样校准数据。"""

    def __init__(self, act_dim: int = 12, act_low: float = -1.0, act_high: float = 1.0):
        self.act_dim = act_dim
        self.act_low = act_low
        self.act_high = act_high

    def act(self, obs: np.ndarray) -> np.ndarray:  # noqa: D401 - 协议兼容
        return np.random.uniform(self.act_low, self.act_high, size=self.act_dim).astype(
            np.float32
        )


class DummyEnv:
    """Lite3 观测空间的简化仿真环境示例。"""

    def __init__(self, obs_dim: int = 108, act_dim: int = 12):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._rng = np.random.default_rng(seed=42)

    def reset(self) -> np.ndarray:
        return self._rng.standard_normal(self.obs_dim).astype(np.float32)

    def step(self, action: np.ndarray):  # noqa: D401 - 简化示例
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        reward = float(self._rng.normal())
        done = False
        info = {}
        return obs, reward, done, info


def make_env() -> DummyEnv:
    return DummyEnv()

