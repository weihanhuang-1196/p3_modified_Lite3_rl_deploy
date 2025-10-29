"""使用 RKNN Toolkit 将 ONNX PPO 模型转换为 RKNN。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from rknn.api import RKNN
except ImportError as exc:  # pragma: no cover - 运行环境无 RKNN 时提示
    raise SystemExit(
        "未检测到 rknn-toolkit2，请先在 RK3588 环境或 x86 转换机安装。"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", type=Path, help="输入的 ONNX 模型路径")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="量化校准数据 (numpy .npy 或 .npz)，内容为 obs 样本",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("policy.rknn"),
        help="导出的 RKNN 模型路径",
    )
    parser.add_argument(
        "--quant-dtype",
        type=str,
        default="asymmetric_affine-u8",
        choices=["asymmetric_affine-u8", "dynamic_fixed-point-i8"],
        help="量化类型，默认使用非对称 UINT8",
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        default=3,
        choices=range(4),
        help="RKNN 优化级别，范围 0-3",
    )
    parser.add_argument(
        "--do-quant", action="store_true", help="是否执行量化 (默认 False)"
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "obs" in data.files:
            arr = data["obs"]
        else:
            arr = data[data.files[0]]
    else:
        arr = data
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return [sample.astype(np.float32) for sample in arr]


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)

    rknn = RKNN(verbose=True)
    rknn.config(
        target_platform="rk3588",
        optimization_level=args.opt_level,
        quantized_dtype=args.quant_dtype,
        output_optimize_level=1,
    )

    print("[INFO] 加载 ONNX...")
    ret = rknn.load_onnx(model=args.onnx.as_posix())
    if ret != 0:
        raise SystemExit("加载 ONNX 失败")

    print("[INFO] 构建 RKNN...")
    ret = rknn.build(do_quantization=args.do_quant, dataset=dataset)
    if ret != 0:
        raise SystemExit("构建 RKNN 失败")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("[INFO] 导出 RKNN...")
    ret = rknn.export_rknn(args.output.as_posix())
    if ret != 0:
        raise SystemExit("导出 RKNN 失败")

    print(f"[INFO] 已生成 RKNN 模型: {args.output}")
    rknn.release()


if __name__ == "__main__":
    main()
