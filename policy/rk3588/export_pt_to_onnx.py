"""将 PPO 策略的 .pt 权重导出为 ONNX 格式。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models import ActorCriticConfig, ActorCriticMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pt", type=Path, help="训练阶段导出的 .pt 权重文件")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("example_actor_config.yaml"),
        help="策略网络的结构配置文件 (YAML)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("policy.onnx"),
        help="导出的 ONNX 文件路径",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="ONNX opset 版本，建议 >=11 以兼容 RKNN ToolKit",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="导出动态 batch 维度，便于后续批量推理",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="在加载权重时开启严格模式，默认兼容缺失/冗余键",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ActorCriticConfig.from_file(args.config)
    model = ActorCriticMLP(config)

    state_dict = torch.load(args.pt, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict)
    if missing:
        print("[WARN] state_dict 缺失键：", missing)
    if unexpected:
        print("[WARN] state_dict 冗余键：", unexpected)

    model.eval()
    dummy = torch.zeros(1, config.obs_dim, dtype=torch.float32)
    input_names = ["obs"]
    output_names = ["action"]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"obs": {0: "batch"}, "action": {0: "batch"}}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output.as_posix(),
        opset_version=args.opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"[INFO] 已导出 ONNX: {args.output}")


if __name__ == "__main__":
    main()
