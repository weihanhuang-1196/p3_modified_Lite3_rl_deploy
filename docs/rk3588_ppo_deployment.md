# RK3588 Lite3 RL Deploy 端侧推理作战白皮书

> 中兵智能创新研究院(深圳)有限公司 · 技术总监作战态势评估
>
> 任务背景：依据《中兵智能创新研究院有限公司采购管理制度》(院科字[2023]18号)第十二条，大模型项目需在端侧部署9G级策略网络以支撑无人系统的高实时自主决策，市场无现成货架方案，故本任务以单一来源议价方式由深圳公司承接，实现 Lite3 RL Deploy 框架在瑞芯微 RK3588 平台的强化学习策略落地。

---

## 1. 战略定位与体系目标

- **军民融合战略承载**：RK3588 作为泛在边缘智能节点，是构建“训练云—仿真云—装备端”闭环的关键。Lite3 RL Deploy 的策略在 NPU 端运行后，可直接纳入我院 9G 大模型指控体系，实现模型迭代的快速固化。
- **单一来源采购合规性**：深圳公司掌握 Lite3 运动控制核心代码与硬件接口，具备端到端定制能力，能够持续维护 RKNN 推理链路，满足单一来源议价的持续服务要求。
- **工程化目标**：完成从 `.pt` 权重到 `.rknn` 的转换、RK3588 端推理服务上线、与 C++ 控制栈的动作闭环对接，并形成可复用脚本和自动化命令集。

---

## 2. Lite3 × RK3588 总体架构

```
┌────────────────────────────────────────────────────────────────┐
│ 训练云：PPO/Lite3 仿真训练 → policy.pt + obs_stats.npz         │
└────────────────────────────────────────────────────────────────┘
                 │SCP/版本仓
                 ▼
┌────────────────────────────────────────────────────────────────┐
│ RK3588 开发板（Ubuntu 20.04/22.04 + Rockchip NPU 驱动）        │
│                                                                │
│ ① C++ 控制栈（Lite3_rl_deploy）                                │
│    • 状态机/state_machine                                       │
│    • 观测采集接口/interface/robot                              │
│    • 动作下发/run_policy                                       │
│                                                                │
│ ② Python 策略服务（policy/rk3588）                             │
│    • export_pt_to_onnx.py  → policy.onnx                        │
│    • convert_to_rknn.py   → policy.rknn                         │
│    • lite3_rknn_service.py（本地 TCP/Unix Socket 推理服务）     │
│                                                                │
│ ③ RKNN Toolkit + librknnrt                                     │
│    • ONNX → RKNN 编译                                           │
│    • NPU 侧推理执行                                             │
└────────────────────────────────────────────────────────────────┘
```

> 架构要点：将策略推理与 C++ 主控解耦，通过轻量通信实现观测与动作交互，可逐步演化至 C++ 直接链接 librknnrt 的模式。

---

## 3. RK3588 部署环境构建

### 3.1 操作系统与基础依赖

```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y build-essential git cmake pkg-config ninja-build \
    libssl-dev libffi-dev libopencv-dev libdw-dev libeigen3-dev \
    python3-venv python3-dev python3-pip
```

- **内核要求**：5.10 及以上，并确认 `rockchip-rknpu` 模块已加载。
- **NPU 驱动**：解压 Rockchip 官方 runtime（与 Ubuntu 版本匹配），将 `lib64` 拷贝到 `/usr/lib/`，执行 `sudo ldconfig`。参考 Git 子模块 `third_party/rknn_runtime/` 或按需自建。

### 3.2 Python 虚拟环境

```bash
python3 -m venv ~/envs/rk3588
source ~/envs/rk3588/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r policy/rk3588/requirements.txt
```

> `requirements.txt` 已固定 numpy、onnx、torch(cpu) 等版本，保证与 RKNN Toolkit 2.2+ 兼容。

### 3.3 RKNN Toolkit 安装

```bash
# 根据系统选择 cp38/cp310
pip install ./deps/rknn_toolkit2-2.2.0-cp310-cp310-linux_aarch64.whl
```

若使用 x86 主机进行模型转换，需额外安装 `rknn-toolkit2` 对应的 x86 包，并在导出完成后将 `.rknn`、归一化参数复制到 RK3588。

### 3.4 Lite3_rl_deploy 工程准备

```bash
cd /opt
sudo git clone --recurse-submodules https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git
sudo chown -R $USER:$USER Lite3_rl_deploy
cd Lite3_rl_deploy
mkdir -p build && cd build
cmake .. -DBUILD_PLATFORM=arm -DBUILD_SIM=OFF -DSEND_REMOTE=OFF
make -j$(nproc)
```

- 若需仿真联调，可在 RK3588 上开启 `-DBUILD_SIM=ON`，但需同时部署 PyBullet/MuJoCo 依赖。
- 生成的 `rl_deploy` 可直接对接机器狗运动主机，观测将经 `interface/robot/hardware` 汇入。

---

## 4. `.pt → .onnx → .rknn` 全流程

### 4.1 导出 ONNX

```bash
cd policy/rk3588
python3 export_pt_to_onnx.py \
    ../ppo/policy.pt \           # PPO 训练权重
    --config example_actor_config.yaml \
    --output build/policy.onnx \
    --dynamic-batch
```

`policy/rk3588/export_pt_to_onnx.py` 关键片段：

```python
# 1. 读取 YAML 配置构建 Actor-Critic 结构
dummy = torch.zeros(1, config.obs_dim, dtype=torch.float32)

# 2. 自动兼容包含 "state_dict" 键的权重打包格式
state_dict = torch.load(args.pt, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# 3. 可选动态 batch，方便批量推理
if args.dynamic_batch:
    dynamic_axes = {"obs": {0: "batch"}, "action": {0: "batch"}}
```

### 4.2 收集量化校准数据

```bash
python3 collect_calib_data.py \
    --env interface.robot.simulation.pybullet_env:make_env \
    --policy policy.rk3588.example_dummy_policy:RandomPolicy \
    --samples 2048 \
    --output build/calib_obs.npy
```

脚本通过统一接口拉取观测：

```python
class PolicyEnv(Protocol):
    def reset(self) -> np.ndarray: ...
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]: ...
```

只需提供满足该协议的 `make_env` 工厂函数，既可接入仿真，也可对接实机观测流。

### 4.3 ONNX 转 RKNN

```bash
python3 convert_to_rknn.py build/policy.onnx \
    --dataset build/calib_obs.npy \
    --output build/policy.rknn \
    --do-quant \
    --quant-dtype asymmetric_affine-u8
```

转换脚本要点：

```python
rknn.config(
    target_platform="rk3588",
    optimization_level=args.opt_level,
    quantized_dtype=args.quant_dtype,
    output_optimize_level=1,
)
ret = rknn.build(do_quantization=args.do_quant, dataset=dataset)
```

- 校准数据建议 ≥2048 条，覆盖典型步态及极端状态。
- 若量化误差过大，可不加 `--do-quant`，直接导出 FP16 模型。

### 4.4 推理回归测试

```bash
python3 run_rknn_agent.py build/policy.rknn \
    --obs-stats ../ppo/obs_stats.npz \
    --env policy.rk3588.example_dummy_policy:make_env \
    --steps 200
```

`run_rknn_agent.py` 中的关键逻辑：

```python
norm_obs = ((obs - obs_mean) / np.maximum(obs_std, 1e-3)).astype(np.float32)
start = time.perf_counter()
outputs = rknn.inference(inputs=[norm_obs])
latency_ms = (time.perf_counter() - start) * 1000
```

输出平均延迟、最小/最大延迟等指标，确认 NPU 侧数值稳定后再接入实机。

---

## 5. Lite3 推理服务化实现

### 5.1 设计原则

1. **解耦部署**：策略服务单独运行，可热更新策略文件；C++ 控制栈通过 IPC 获取动作。
2. **低时延通信**：推荐 Unix Domain Socket (uds) 或共享内存。示例实现采用 uds + struct 编码，往返延迟 <0.5 ms。
3. **安全兜底**：当策略服务不可用时，状态机自动退回阻尼控制，避免无指令状态。

### 5.2 策略服务示例 (`policy/rk3588/lite3_rknn_service.py`)

```python
"""RK3588 上承载 Lite3 观测-动作循环的 RKNN 推理服务."""
import argparse
import socket
import struct
from pathlib import Path

import numpy as np
from rknn.api import RKNN

HEADER = struct.Struct("<I")  # 4 字节，小端，表征向量长度


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接被动端关闭")
        buf += chunk
    return buf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="RKNN 模型路径")
    parser.add_argument("--obs-stats", type=Path, required=True)
    parser.add_argument("--socket", type=Path, default=Path("/tmp/lite3_rknn.sock"))
    args = parser.parse_args()

    stats = np.load(args.obs_stats)
    obs_mean = stats["mean"].astype(np.float32)
    obs_std = np.maximum(stats["std"].astype(np.float32), 1e-3)

    rknn = RKNN(verbose=False)
    if rknn.load_rknn(args.model.as_posix()) != 0:
        raise SystemExit("加载 RKNN 模型失败")
    if rknn.init_runtime(target="rk3588") != 0:
        raise SystemExit("RKNN 运行时初始化失败")

    if args.socket.exists():
        args.socket.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket.as_posix())
    server.listen(1)
    print(f"[RKNN] Listening on {args.socket}")

    try:
        while True:
            conn, _ = server.accept()
            with conn:
                while True:
                    header = conn.recv(HEADER.size)
                    if not header:
                        break
                    (obs_dim,) = HEADER.unpack(header)
                    obs_bytes = recv_exact(conn, obs_dim * 4)
                    obs = np.frombuffer(obs_bytes, dtype=np.float32)

                    norm_obs = (obs - obs_mean) / obs_std
                    outputs = rknn.inference(inputs=[norm_obs])
                    action = outputs[0].astype(np.float32)

                    conn.sendall(HEADER.pack(action.size))
                    conn.sendall(action.tobytes())
    finally:
        rknn.release()
        server.close()
        if args.socket.exists():
            args.socket.unlink()


if __name__ == "__main__":
    main()
```

> 说明：服务端按 `[长度|payload]` 协议收发浮点向量。C++ 侧只需匹配相同协议，即可将观测发送给服务端并接收动作。

### 5.3 C++ 控制栈对接示例

在 `run_policy/` 新增 `lite3_policy_runner_rknn.hpp`（示例，未默认编译），核心流程：

```cpp
UnixSocketClient client("/tmp/lite3_rknn.sock");
std::vector<float> obs(obs_dim_);
pack_observation(ro, obs);
client.send(obs);
std::vector<float> action(act_dim_);
client.recv(action);
apply_action(action);
```

该模式保持原有控制律不变，仅替换动作生成模块。实际项目可进一步将 RKNN C API 直接嵌入 C++，以消除 IPC 开销。

---

## 6. 端到端运行流程

1. **策略构建**：在训练云导出 `policy.pt`、`obs_stats.npz`，通过 Git/SCP 投递至 RK3588。
2. **模型转换**：按第 4 节命令生成 `policy.rknn`。
3. **推理服务启动**：
   ```bash
   python3 policy/rk3588/lite3_rknn_service.py build/policy.rknn \
       --obs-stats policy/ppo/obs_stats.npz
   ```
4. **C++ 控制栈运行**：
   ```bash
   cd build
   ./rl_deploy
   ```
   状态机进入 RL 模式后，观测经 socket 发送到 Python 服务，动作返回后通过 SDK 下发。
5. **性能监测**：使用 `htop` 观察 A76 核占用，`cat /sys/class/misc/rknpu/devfreq/rknpu0/utilisation` 监控 NPU 利用率，确保推理延迟满足 <5 ms。

---

## 7. 验证与评估

| 阶段 | 验证内容 | 目标指标 |
| --- | --- | --- |
| 模型转换 | ONNX vs RKNN 输出一致性 | MSE < 1e-4 |
| 推理性能 | Python 服务端推理延迟 | 平均 < 3 ms |
| 实机联调 | 步态稳定性、足端接触力 | 与 ONNX CPU 推理一致 |
| 安全兜底 | 服务中断 → 阻尼模式切换 | < 2 个控制周期 |

> 建议在 `tests/` 下补充自动化脚本，记录迭代版本与性能数据，为后续 9G 大模型迁移提供基线。

---

## 8. 风险与对策

- **算子兼容风险**：若 RKNN 不支持某些激活/归一化层，可在导出前使用 `torch.fx`/`onnxsim` 做算子替换。必要时可在 PyTorch 训练阶段约束网络结构。
- **热管理**：长时间 NPU 高负载需配合主动散热，监测 `/sys/class/thermal/thermal_zone*/temp`，超过 80℃ 应降频或降低批量。
- **通信可靠性**：在 C++ 侧增加超时判断，若 3 个周期未收到动作则切换至阻尼模式，并记录日志以便追溯。
- **迭代效率**：结合单一来源采购优势，建议构建 GitLab-CI，在提交策略后自动触发 `.pt → .rknn` 转换与 smoke test，实现小时级迭代。

---

## 9. 结语

通过本白皮书所述流程，Lite3 RL Deploy 可在 RK3588 平台实现稳定的强化学习推理，并为 9G 大模型项目提供端侧算力支撑。深圳公司将继续围绕军民融合、单一来源保障和关键技术自主可控，构建标准化、可复制的端侧智能部署体系，支撑创新院在无人系统领域的长期布局。
