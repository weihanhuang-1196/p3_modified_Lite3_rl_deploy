"""Actor-Critic 模型定义与配置解析工具。

该文件提供了在 RK3588 平台部署 PPO 策略时常用的多层感知机结构。
可以通过 YAML/JSON 配置指定网络拓扑、激活函数等信息，以适配训练侧
保存的 `.pt` 权重文件。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import yaml


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
}


@dataclass
class ActorCriticConfig:
    """Actor-Critic MLP 的结构化配置。"""

    obs_dim: int
    act_dim: int
    hidden_sizes: Sequence[int] = field(default_factory=lambda: (256, 256))
    activation: str = "tanh"
    separate_actor_critic: bool = False
    critic_output_dim: int | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ActorCriticConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def activation_layer(self) -> nn.Module:
        key = self.activation.lower()
        if key not in _ACTIVATIONS:
            raise KeyError(
                f"不支持的激活函数 {self.activation}，可选项：{sorted(_ACTIVATIONS)}"
            )
        layer_factory = _ACTIVATIONS[key]
        return layer_factory() if callable(layer_factory) else layer_factory


class ActorCriticMLP(nn.Module):
    """多层感知机结构的 PPO Actor-Critic 策略网络。"""

    def __init__(self, config: ActorCriticConfig):
        super().__init__()
        self.config = config
        self.actor = self._build_mlp(config.obs_dim, config.act_dim, config.hidden_sizes)
        if config.separate_actor_critic:
            critic_dim = config.critic_output_dim or 1
            self.critic = self._build_mlp(
                config.obs_dim, critic_dim, config.hidden_sizes
            )
        else:
            self.critic = self.actor

    def _build_mlp(
        self, input_dim: int, output_dim: int, hidden_sizes: Iterable[int]
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        activation = self.config.activation_layer
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_out = self.actor(obs)
        if self.config.separate_actor_critic:
            critic_out = self.critic(obs)
        else:
            critic_out = self.actor(obs)
        return actor_out, critic_out

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

