"""RK3588 上运行 PPO 策略的推理脚本示例。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    from rknn.api import RKNN
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要在 RK3588 上安装 rknn-toolkit2 才能执行该脚本") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="RKNN 模型路径")
    parser.add_argument("--obs-stats", type=Path, required=True, help="观测归一化文件")
    parser.add_argument(
        "--steps", type=int, default=100, help="运行步数，便于评估推理性能"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="policy.rk3588.example_dummy_policy:make_env",
        help="环境工厂函数，格式 module:factory",
    )
    return parser.parse_args()


def load_env(entry: str):
    module_name, _, factory = entry.partition(":")
    module = __import__(module_name, fromlist=[factory])
    factory_fn = getattr(module, factory or "make_env")
    return factory_fn()


def load_obs_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(path)
    return stats["mean"].astype(np.float32), stats["std"].astype(np.float32)


def main() -> None:
    args = parse_args()
    env = load_env(args.env)
    obs_mean, obs_std = load_obs_stats(args.obs_stats)

    rknn = RKNN(verbose=False)
    if rknn.load_rknn(args.model.as_posix()) != 0:
        raise SystemExit("加载 RKNN 模型失败")
    if rknn.init_runtime(target="rk3588") != 0:
        raise SystemExit("初始化 RKNN 运行时失败")

    obs = env.reset()
    latencies: list[float] = []

    for step in range(args.steps):
        norm_obs = ((obs - obs_mean) / np.maximum(obs_std, 1e-3)).astype(np.float32)
        start = time.perf_counter()
        outputs = rknn.inference(inputs=[norm_obs])
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        action = outputs[0]
        obs, reward, done, info = env.step(action)
        print(
            f"step={step:04d} latency={latency_ms:7.3f} ms reward={reward:+.3f} done={done}"
        )
        if done:
            obs = env.reset()

    print(
        f"[INFO] 推理统计 -> mean={np.mean(latencies):.3f} ms, min={np.min(latencies):.3f} ms, max={np.max(latencies):.3f} ms"
    )
    rknn.release()


if __name__ == "__main__":
    main()
