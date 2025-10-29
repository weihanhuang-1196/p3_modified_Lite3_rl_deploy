"""从仿真或实机运行中收集观测量作为量化校准数据。"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from types import ModuleType
from typing import Iterable, Protocol

import numpy as np


class PolicyEnv(Protocol):
    """与 Lite3 接口契合的最小环境协议。"""

    def reset(self) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...


class PolicyModule(Protocol):
    """策略模块需要暴露 act(obs) -> action 接口。"""

    def act(self, obs: np.ndarray) -> np.ndarray: ...


def load_env(entry: str) -> PolicyEnv:
    module_name, _, factory = entry.partition(":")
    module: ModuleType = importlib.import_module(module_name)
    factory_fn = getattr(module, factory or "make_env")
    env = factory_fn()
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="环境工厂函数，格式 module.submodule:make_env",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="policy.rk3588.example_dummy_policy:RandomPolicy",
        help="用于采样动作的数据策略，默认使用随机策略示例",
    )
    parser.add_argument("--samples", type=int, default=1024, help="采样数量")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("calib_obs.npy"),
        help="输出 numpy 文件路径",
    )
    return parser.parse_args()


def load_policy(entry: str) -> PolicyModule:
    module_name, _, cls_name = entry.partition(":")
    module: ModuleType = importlib.import_module(module_name)
    policy_cls = getattr(module, cls_name or "RandomPolicy")
    return policy_cls()


def sample_observations(env: PolicyEnv, policy: PolicyModule, n: int) -> Iterable[np.ndarray]:
    obs = env.reset()
    for _ in range(n):
        action = policy.act(obs)
        obs, *_rest = env.step(action)
        yield obs


def main() -> None:
    args = parse_args()
    env = load_env(args.env)
    policy = load_policy(args.policy)

    buffer = np.stack(list(sample_observations(env, policy, args.samples))).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.as_posix(), buffer)
    print(f"[INFO] 已保存量化校准数据 {args.output}, shape={buffer.shape}")


if __name__ == "__main__":
    main()
