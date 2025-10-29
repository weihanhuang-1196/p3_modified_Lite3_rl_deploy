"""生成演示用的随机初始化策略权重与归一化统计。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models import ActorCriticConfig, ActorCriticMLP


def main() -> None:
    config = ActorCriticConfig.from_file(
        Path(__file__).with_name("example_actor_config.yaml")
    )
    model = ActorCriticMLP(config)
    ckpt_path = Path(__file__).with_name("demo_policy.pt")
    torch.save(model.state_dict(), ckpt_path.as_posix())
    print(f"[INFO] 已生成示例权重: {ckpt_path}")

    obs_mean = np.zeros(config.obs_dim, dtype=np.float32)
    obs_std = np.ones(config.obs_dim, dtype=np.float32)
    stats_path = Path(__file__).with_name("demo_obs_stats.npz")
    np.savez(stats_path.as_posix(), mean=obs_mean, std=obs_std)
    print(f"[INFO] 已生成观测归一化文件: {stats_path}")


if __name__ == "__main__":
    main()
