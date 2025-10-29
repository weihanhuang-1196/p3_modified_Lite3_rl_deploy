"""Lite3 RL Deploy 在 RK3588 上承载 PPO 推理的 Unix Socket 服务。"""
from __future__ import annotations

import argparse
import socket
import struct
from pathlib import Path

import numpy as np

try:  # pragma: no cover - 部署设备需预装 rknn-toolkit2
    from rknn.api import RKNN
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要在设备上安装 rknn-toolkit2 才能运行该服务") from exc

HEADER = struct.Struct("<I")  # 小端无符号整型，表示随后的向量长度


def recv_exact(conn: socket.socket, size: int) -> bytes:
    """阻塞读取 size 字节数据，直到满足或连接关闭。"""
    buffer = bytearray()
    while len(buffer) < size:
        chunk = conn.recv(size - len(buffer))
        if not chunk:
            raise ConnectionError("连接异常断开")
        buffer.extend(chunk)
    return bytes(buffer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="RKNN 模型路径")
    parser.add_argument("--obs-stats", type=Path, required=True, help="观测归一化参数 npz")
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path("/tmp/lite3_rknn.sock"),
        help="Unix Domain Socket 地址 (默认 /tmp/lite3_rknn.sock)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stats = np.load(args.obs_stats)
    obs_mean = stats["mean"].astype(np.float32)
    obs_std = np.maximum(stats["std"].astype(np.float32), 1e-3)

    rknn = RKNN(verbose=False)
    if rknn.load_rknn(args.model.as_posix()) != 0:
        raise SystemExit("加载 RKNN 模型失败")
    if rknn.init_runtime(target="rk3588") != 0:
        raise SystemExit("初始化 RKNN 运行时失败")

    sock_path = args.socket
    if sock_path.exists():
        sock_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path.as_posix())
    server.listen(1)
    print(f"[RKNN] Lite3 推理服务监听 {sock_path}")

    try:
        while True:
            conn, _ = server.accept()
            print("[RKNN] 控制端连接已建立")
            with conn:
                while True:
                    header = conn.recv(HEADER.size)
                    if not header:
                        break
                    (obs_len,) = HEADER.unpack(header)
                    obs_bytes = recv_exact(conn, obs_len * 4)
                    obs = np.frombuffer(obs_bytes, dtype=np.float32)

                    norm_obs = (obs - obs_mean) / obs_std
                    outputs = rknn.inference(inputs=[norm_obs])
                    action = outputs[0].astype(np.float32)

                    conn.sendall(HEADER.pack(action.size))
                    conn.sendall(action.tobytes())
    except KeyboardInterrupt:
        print("[RKNN] 服务终止")
    finally:
        rknn.release()
        server.close()
        if sock_path.exists():
            sock_path.unlink()


if __name__ == "__main__":
    main()
