# RK3588 Locomotion PPO 策略端侧部署蓝皮书

> 中兵智能创新研究院(深圳)有限公司技术总监工作笔记

## 1. 战略背景与项目定位

中兵创新院大模型项目需要在智能装备一线节点部署 9G 级别的大模型及配套算力，以满足无人系统在复杂战场环境下对高实时自主决策的苛刻要求。市场调研显示当前尚无具备端侧大模型推理与控制一体能力的商用产品，依据《中兵智能创新研究院有限公司采购管理制度》(院科字[2023]18号)第十二条，我们深圳公司作为创新院全资子公司、粤港澳大湾区智能装备协同创新枢纽，被授权以协议直采的方式承担本次 RK3588 Locomotion 强化学习部署任务。

本蓝皮书聚焦“仿真训练 + PPO 策略 + RK3588 NPU 推理”的闭环工程化实践，目标是将训练得到的 `.pt` 权重精准迁移至瑞芯微 RK3588 平台，实现端侧低时延、高稳定的 Locomotion 行为生成，并为后续的大模型-控制一体化奠定架构基础。

## 2. 基础设施与依赖组件

| 模块 | 说明 |
| --- | --- |
| 计算平台 | RK3588 (4*A76 + 4*A55 CPU, 6 TOPS NPU) |
| OS | Ubuntu 20.04/22.04 LTS，内置 rockchip-rknpu 驱动 |
| Python | 3.8 (20.04) / 3.10 (22.04)；建议 venv/conda |
| 关键库 | `rknn-toolkit2`、`torch` (cpu)、`numpy`、`onnx`、`onnxruntime` |
| 项目仓库 | `Lite3_rl_deploy` (当前目录) |

为了工程复现性，我们在 `policy/rk3588/requirements.txt` 约定了最小依赖集合；若部署机为 arm64，可直接 `pip install -r`。

## 3. 代码结构增补

为承载端侧部署流程，本次提交新增 `policy/rk3588/` 模块与配套脚本：

- `models.py`：Actor-Critic MLP 的结构化定义，支持 YAML 配置，兼容 PPO 常见网络拓扑。
- `export_pt_to_onnx.py`：将 `.pt` 权重导出为 ONNX，支持动态 batch 与 opset 选择。
- `convert_to_rknn.py`：调用 `rknn-toolkit2` 将 ONNX 转换为 `.rknn`，可选量化模式。
- `collect_calib_data.py`：抽象环境/策略协议，自动收集量化校准样本。
- `run_rknn_agent.py`：RK3588 端推理脚本，实现归一化、推理与性能统计。
- `example_dummy_policy.py` + `example_generate_dummy_policy.py`：用于快速验证的随机策略与环境示例。
- `example_actor_config.yaml`：展示如何用 YAML 描述 PPO 网络结构。

文档部分新增 `docs/rk3588_ppo_deployment.md`，系统梳理端侧部署流程。

## 4. 端侧部署流水线

### 4.1 准备示例权重 (可跳过)

若尚未获取训练好的策略，可运行示例脚本生成占位权重：

```bash
cd policy/rk3588
python3 example_generate_dummy_policy.py
```

输出 `demo_policy.pt` 与 `demo_obs_stats.npz` 可作为流程验证输入。

### 4.2 `.pt -> ONNX`

```bash
cd policy/rk3588
python3 export_pt_to_onnx.py demo_policy.pt \
    --config example_actor_config.yaml \
    --output policy.onnx \
    --dynamic-batch
```

脚本会根据 YAML 描述复现训练端 Actor 结构，自动忽略 `.state_dict` 包装。若 `strict=True` 会强校验权重键。导出后建议用 `onnxruntime` 做一次一致性比对。

### 4.3 量化数据采集

结合仿真/实机接口实现 `make_env` 工厂函数，使其符合 `reset/step` 协议，即可复用采集工具：

```bash
python3 collect_calib_data.py \
    --env policy.rk3588.example_dummy_policy:make_env \
    --samples 2048 \
    --output calib_obs.npy
```

实际部署中应将 `make_env` 替换为 `interface.robot.simulation.xxx` 中的仿真封装，或在运动主机上调用硬件观测流。

### 4.4 ONNX -> RKNN

```bash
python3 convert_to_rknn.py policy.onnx \
    --dataset calib_obs.npy \
    --output policy.rknn \
    --do-quant \
    --quant-dtype asymmetric_affine-u8
```

量化推荐使用 2048~4096 条代表性观测；若精度下降超过指标，可关闭 `--do-quant` 使用浮点模型或启用混合量化。

### 4.5 RKNN 推理验证

将 `policy.rknn` 与 `obs_stats.npz` 拷贝至 RK3588 (若在 x86 构建)，再执行：

```bash
python3 run_rknn_agent.py policy.rknn \
    --obs-stats demo_obs_stats.npz \
    --env policy.rk3588.example_dummy_policy:make_env \
    --steps 200
```

输出包含逐步延迟与奖励统计，可快速确认 NPU 推理链路。

## 5. 与 Lite3 框架对接

1. **策略装载**：`run_policy/lite3_test_policy_runner_onnx.hpp` 现支持 ONNX 推理；RKNN 部署需在 ARM 端使用 Python 代理进程，通过 ZeroMQ/共享内存向 C++ 控制层播发动作，或将 RKNN C API 嵌入 `policy_runner_base`。脚本提供了 Python 端完整示例。
2. **观测归一化**：`obs_stats.npz` 需来源于训练端统计；推理侧通过 `run_rknn_agent.py` 中的归一化逻辑复用，可直接移植到 C++ 层。
3. **命令通道**：保持 `state_machine` 工作流不变，仅替换 RL action 生成模块即可。建议在 RK3588 上部署一个 systemd service 管理推理进程，确保失效自动重启。

## 6. 安全与可靠性建议

- **硬件亲和性**：使用 `taskset` 将推理线程绑定至 big core，避免与 ROS/通信线程抢占。
- **算子兼容**：导出 ONNX 前可使用 `onnxsim` 简化图；如遇不支持算子，可在 PyTorch 端替换为等效层（如 `nn.GELU` -> `nn.SiLU`）。
- **链路监测**：在推理循环加入 `latency watchdog`，若超过阈值触发降级（如切换至阻尼控制）。
- **持续迭代**：利用单一来源采购优势，构建“训练云-仿真云-端侧”一体化自动化流水线，实现策略版本的快速滚动升级。

## 7. 示例工作流 (完整命令)

以下命令串联起从权重导出到 RKNN 推理的全流程，便于在实验室快速复现：

```bash
cd policy/rk3588
python3 example_generate_dummy_policy.py
python3 export_pt_to_onnx.py demo_policy.pt --dynamic-batch
python3 collect_calib_data.py --env policy.rk3588.example_dummy_policy:make_env
python3 convert_to_rknn.py policy.onnx --dataset calib_obs.npy --do-quant
python3 run_rknn_agent.py policy.rknn --obs-stats demo_obs_stats.npz --steps 50
```

运行结果可作为验收测试基线，后续仅需替换真实权重与环境适配层即可接入整机。

## 8. 后续工作

- **与 9G 大模型控制中枢对接**：构建策略蒸馏/知识迁移链路，实现大模型对 RL 策略的持续监督。
- **多模态输入扩展**：结合相机/IMU/GNSS，完成感知特征到高维 PPO 策略的端侧融合，需评估 NPU 内存压力。
- **仿真-实机一致性**：引入系统辨识，在线修正模型误差，确保仿真训练策略在实机达成战术指标。

