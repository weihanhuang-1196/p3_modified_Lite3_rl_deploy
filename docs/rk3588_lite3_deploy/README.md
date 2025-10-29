# RK3588 Lite3 RL Deploy 作战蓝皮书


---

## 0. 文档矩阵

| 文档 | 用途 | 对应脚本/目录 |
| --- | --- | --- |
| 本 README | 全局概览、环境搭建、链路说明 | `policy/rk3588/` |
| [`deployment_pipeline.md`](./deployment_pipeline.md) | `.pt→.onnx→.rknn` 转换细节与排错 | `export_pt_to_onnx.py`, `convert_to_rknn.py`, `collect_calib_data.py` |
| [`runtime_service.md`](./runtime_service.md) | 端侧推理服务、Lite3 控制栈对接流程 | `lite3_rknn_service.py`, `run_rknn_agent.py` |

---

## 1. 目标

1.RK3588 端侧强化学习推理。
---

## 2. 仓库新增目录与脚本

```
policy/rk3588/
├── __init__.py
├── collect_calib_data.py      # 量化校准样本采集
├── convert_to_rknn.py         # ONNX→RKNN 编译脚本
├── example_actor_config.yaml  # Actor-Critic 超参数样例
├── example_dummy_policy.py    # 校准阶段的随机策略示例
├── example_generate_dummy_policy.py
├── export_pt_to_onnx.py       # .pt→.onnx 导出脚本
├── lite3_rknn_service.py      # RKNN Unix Socket 推理服务
├── models.py                  # Lite3 PPO Actor/Value 网络结构
├── requirements.txt           # Toolkit 与 Python 依赖锁定
└── run_rknn_agent.py          # 本地推理验证 Demo
```

> 所有脚本默认在 RK3588 上运行；若于 x86 主机转换 `.rknn`，只需在命令中添加 `--target rk3588`，并将导出结果下发到板端。

---

## 3. 基础环境搭建

### 3.1 操作系统基线

- **硬件**：Rockchip RK3588 NPU SoC，≥8GB LPDDR4x。
- **系统建议**：Ubuntu 20.04/22.04 (kernel ≥5.10)，已集成 `rockchip-rknpu` 模块。
- **驱动部署**：
  ```bash
  wget https://github.com/rockchip-linux/rknpu2/raw/master/runtime/Linux/librknn_api/aarch64/ubuntu22.04/rknn_api_linux_aarch64_ubuntu22.04.tar.gz
  tar -xzf rknn_api_linux_aarch64_ubuntu22.04.tar.gz
  cd rknn_api_linux_aarch64_ubuntu22.04
  sudo cp -r lib64/* /usr/lib/
  sudo cp -r include/* /usr/include/
  sudo ldconfig
  ```
  对于内核未预装驱动的场景，执行 `rknpu2/driver/Linux Kernel Driver` 下的 `make && sudo make install`，并通过 `lsmod | grep rknpu` 验证加载状态。

### 3.2 Python 虚拟环境

```bash
python3 -m venv ~/envs/rk3588
source ~/envs/rk3588/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r policy/rk3588/requirements.txt
```

- `requirements.txt` 锁定 `numpy==1.23.5`, `onnx==1.13.1`, `torch==1.13.1+cpu`, `rknn_toolkit2-2.2.0` 等版本，避免算子兼容性问题。
- 若离线部署，可提前将 whl 包放入 `deps/` 并使用 `pip install ./deps/<package>.whl`。

### 3.3 Lite3 控制栈编译

```bash
cd Lite3_rl_deploy
mkdir -p build && cd build
cmake .. -DBUILD_PLATFORM=arm -DBUILD_SIM=OFF -DSEND_REMOTE=OFF
make -j$(nproc)
```

- `-DBUILD_SIM=ON` 可在板上启用仿真联调（需额外安装 PyBullet/MuJoCo）。
- 生成的 `rl_deploy` 将通过 `interface/robot/hardware` 采集观测并下发关节动作。

---

## 4. 部署流水线概览

1. **策略资产入库**：将训练产出的 `policy.pt` 与归一化统计 `obs_stats.npz` 上传至 `policy/rk3588/inputs/`（目录可自建）。
2. **ONNX 导出**：运行 `export_pt_to_onnx.py` 根据 YAML 配置构建网络并导出 `policy.onnx`。
3. **量化校准**：通过 `collect_calib_data.py` 抽样 1k~4k 观测，生成 `calib_obs.npy`。
4. **RKNN 编译**：`convert_to_rknn.py` 完成量化、算子检查并产出 `policy.rknn`。
5. **推理服务上线**：`lite3_rknn_service.py` 在 RK3588 上启动 Unix Socket 服务，提供 `<obs, action>` 映射。
6. **C++ 控制闭环**：`run_rknn_agent.py` 或自有控制栈读取服务输出，实现步态控制闭环。

> 每一阶段的详细参数、可选项与问题排查请查阅对应子文档。

---

## 5. 快速验证脚本

### 5.1 导出 PPO 策略为 ONNX

```bash
cd policy/rk3588
python3 export_pt_to_onnx.py \
    ./inputs/policy.pt \
    --config example_actor_config.yaml \
    --output build/policy.onnx \
    --dynamic-batch
```

关键实现：

```python
with open(args.config) as f:
    cfg = yaml.safe_load(f)
model = ActorCritic(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], hidden_sizes=cfg["hidden_sizes"])
state_dict = torch.load(args.pt, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)
model.eval()
```

> `--dynamic-batch` 将 `obs` 与 `action` 维度声明为动态 batch，有利于批量推理。

### 5.2 收集量化数据

```bash
python3 collect_calib_data.py \
    --env interface.robot.simulation.pybullet_env:make_env \
    --policy policy.rk3588.example_dummy_policy:RandomPolicy \
    --samples 2048 \
    --output build/calib_obs.npy
```

要点：

- 通过 `模块路径:函数名` 的动态导入机制，替换为实机采集或仿真环境皆可。
- `RandomPolicy` 仅作为示例，生产环境应替换为策略网络或控制器以覆盖真实分布。

### 5.3 编译 RKNN

```bash
python3 convert_to_rknn.py build/policy.onnx \
    --dataset build/calib_obs.npy \
    --output build/policy.rknn \
    --target rk3588 \
    --do-quant \
    --quant-dtype asymmetric_affine-u8
```

内部流程：

```python
rknn = RKNN(verbose=True)
rknn.config(target_platform='rk3588', optimization_level=3, output_optimize_level=1,
            quantized_dtype=args.quant_dtype, mean_values=None, std_values=None)
rknn.load_onnx(model=args.onnx)
rknn.build(do_quantization=args.do_quant, dataset=args.dataset)
rknn.export_rknn(args.output)
```

若 `do_quant` 关闭，将输出浮点 RKNN，便于快速验证算子支持性。

### 5.4 启动推理服务

```bash
python3 lite3_rknn_service.py \
    --model build/policy.rknn \
    --obs-stats inputs/obs_stats.npz \
    --socket /tmp/lite3_policy.sock \
    --warmup 16
```

服务逻辑：

```python
obs = np.frombuffer(client.recv(expected_bytes), dtype=np.float32)
obs = (obs - self.obs_mean) / np.maximum(self.obs_std, 1e-3)
outputs = self.rknn.inference(inputs=[obs])
client.sendall(outputs[0].astype(np.float32).tobytes())
```

- 默认 Unix Socket，可通过 `--tcp` 切换到 TCP 服务。
- `--warmup` 预热多次推理，使 NPU 频点稳定。

### 5.5 本地环路测试

```bash
python3 run_rknn_agent.py \
    --model build/policy.rknn \
    --obs-stats inputs/obs_stats.npz \
    --env interface.robot.simulation.pybullet_env:make_env \
    --steps 512
```

该脚本直接在 Python 内完成观测预处理与动作采样，便于验证 RKNN 推理输出是否与 ONNX 基线一致。

---

## 6. 性能与验收指标

| 指标 | 验收标准 | 工程措施 |
| --- | --- | --- |
| NPU 推理延迟 | 单步 < 5 ms (batch=1) | `optimization_level=3`、量化 U8、预热 16 次 |
| 观测预处理延迟 | < 1 ms | NumPy 向量化、必要时转写 C++/NEON |
| 动作一致性 | RKNN vs ONNX MAE < 1e-3 | 增加校准样本、必要时混合量化 |
| 稳态功耗 | < 10 W | NPU 动态频控、关闭多余服务 |

验收流程：

1. 通过 `compare_outputs.py`（可自定义）对比 RKNN 与 ONNX 输出误差。
2. 记录服务启动日志与推理时延，形成测试报告归档。
3. 在 Lite3 实机上完成连续 10 分钟步态稳定性验证，记录 IMU、足端力矩指标。

---

## 7. 维护与升级建议

- **模型迭代**：每次策略迭代需同步更新 `policy.pt` 与 `obs_stats.npz`，重走导出与编译流程；建议编写 CI 任务自动化执行。
- **Toolkit 升级**：Rockchip 发布新版本时，先在仿真环境验证算子支持性，再在板端灰度升级。
- **多模型管理**：可将多个 `.rknn` 存入 `/opt/models/<version>/`，通过服务参数切换版本，并记录版本号以支持回滚。
- **安全加固**：在军工场景部署时，请结合 SELinux/AppArmor 对 Unix Socket、TCP 端口施加访问控制，并监控推理服务运行态势。

---
