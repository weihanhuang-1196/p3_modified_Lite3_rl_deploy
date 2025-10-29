# Lite3 RK3588 部署流水线代码总览


> **任务场景：**基于 Lite3 Deploy 框架在 RK3588 平台闭环运行强化学习策略（`policy.pt`），实现控制指令与感知数据的实时互联互通。

本索引文件聚焦 RK3588 端到端推理链路的全部核心代码资产，拆解 `policy.pt → policy.onnx → policy.rknn → 控制栈动作执行` 的流水线，实现研发、量化、部署、通信与数据闭环可追溯。每个章节都列出关键源码文件、职责边界与相互依赖，并给出结构化图示与排查要点。

---

## 1. 策略资产建模层（Python）

| 阶段 | 源码文件 | 作用 | 备注 |
| --- | --- | --- | --- |
| 模型结构定义 | [`policy/rk3588/models.py`](../../policy/rk3588/models.py) | 定义 Actor-Critic MLP 拓扑，统一 obs/action 维度、激活函数 | 与 YAML 配置字段一一对应，确保导出/推理一致 |
| 网络配置 | [`policy/rk3588/example_actor_config.yaml`](../../policy/rk3588/example_actor_config.yaml) | 记录 `obs_dim`、`act_dim`、隐藏层、激活等参数 | 训练、导出、推理三端共用 |
| PT→ONNX | [`policy/rk3588/export_pt_to_onnx.py`](../../policy/rk3588/export_pt_to_onnx.py) | 加载 `.pt` 权重、重建模型、导出 ONNX | 支持动态 batch、state_dict 兼容加载 |

### 1.1 `export_pt_to_onnx.py` 关键实现
```python
config = ActorCriticConfig.from_file(args.config)
model = ActorCriticMLP(config)
state_dict = torch.load(args.pt, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
missing, unexpected = model.load_state_dict(state_dict, strict=args.strict)
model.eval()
dummy = torch.zeros(1, config.obs_dim, dtype=torch.float32)
torch.onnx.export(
    model,
    dummy,
    args.output.as_posix(),
    opset_version=args.opset,
    input_names=["obs"],
    output_names=["action"],
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}} if args.dynamic_batch else None,
)
```

**接口对接要点：**
- `config.obs_dim` 与 `config.act_dim` 必须与 C++ 端 `Lite3TestPolicyRunnerONNX` 固化的维度一致。
- `state_dict` 支持 `checkpoint['state_dict']` 与纯权重两种结构，便于兼容各训练框架。

---

## 2. 量化与 RKNN 编译层（Python）

| 阶段 | 源码文件 | 作用 | 备注 |
| --- | --- | --- | --- |
| 量化数据采集 | [`policy/rk3588/collect_calib_data.py`](../../policy/rk3588/collect_calib_data.py) | 采集观测样本生成 `calib_obs.npy` | 兼容仿真/实机数据源 |
| ONNX→RKNN | [`policy/rk3588/convert_to_rknn.py`](../../policy/rk3588/convert_to_rknn.py) | 配置 RKNN ToolKit，完成量化、编译、导出 | 支持浮点/量化模型 |
| 板端推理验证 | [`policy/rk3588/run_rknn_agent.py`](../../policy/rk3588/run_rknn_agent.py) | 直接调用 RKNN 推理接口，验证性能与输出 | 可接入仿真环境快速回归 |

### 2.1 `convert_to_rknn.py` 关键流程
```python
rknn = RKNN(verbose=args.verbose)
rknn.config(
    target_platform=args.target,
    optimization_level=args.optimization_level,
    output_optimize_level=args.output_optimize_level,
    quantized_dtype=args.quant_dtype,
)
rknn.load_onnx(model=args.onnx)
if args.do_quant:
    rknn.build(do_quantization=True, dataset=args.dataset)
else:
    rknn.build(do_quantization=False)
rknn.export_rknn(args.output)
```

**部署注意事项：**
- `--dataset` 必须指向 `collect_calib_data.py` 生成的观测样本，保证量化精度。
- 量化失败时，可临时关闭 `--do-quant` 输出浮点 RKNN，以定位算子兼容性问题。

---

## 3. RKNN 推理服务层（Python）

| 组件 | 源码文件 | 职责 | 通信接口 |
| --- | --- | --- | --- |
| 推理服务 | [`policy/rk3588/lite3_rknn_service.py`](../../policy/rk3588/lite3_rknn_service.py) | 载入 `policy.rknn` 与 `obs_stats.npz`，提供 Unix Socket 推理服务 | 默认 `/tmp/lite3_rknn.sock` |
| 性能回放 | [`policy/rk3588/run_rknn_agent.py`](../../policy/rk3588/run_rknn_agent.py) | 离线调用 RKNN 模型，统计延迟并验证策略行为 | 本地运行 |

### 3.1 `lite3_rknn_service.py` 核心代码
```python
stats = np.load(args.obs_stats)
obs_mean = stats["mean"].astype(np.float32)
obs_std = np.maximum(stats["std"].astype(np.float32), 1e-3)

rknn = RKNN(verbose=False)
rknn.load_rknn(args.model.as_posix())
rknn.init_runtime(target="rk3588")

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(args.socket.as_posix())
server.listen(1)

while True:
    conn, _ = server.accept()
    with conn:
        header = conn.recv(HEADER.size)
        (obs_len,) = HEADER.unpack(header)
        obs_bytes = recv_exact(conn, obs_len * 4)
        obs = np.frombuffer(obs_bytes, dtype=np.float32)
        norm_obs = (obs - obs_mean) / obs_std
        outputs = rknn.inference(inputs=[norm_obs])
        action = outputs[0].astype(np.float32)
        conn.sendall(HEADER.pack(action.size))
        conn.sendall(action.tobytes())
```

**链路协定：**
- 所有观测/动作均采用 `float32`，通过 `<长度, payload>` 格式传输，确保多态扩展。
- 默认 `obs_len=obs_dim`，控制栈需与 YAML 配置保持一致。

---

## 4. Lite3 控制栈适配层（C++)

| 模块 | 源码文件 | 职责 | 描述 |
| --- | --- | --- | --- |
| 状态机入口 | [`state_machine/rl_control_state_onnx.hpp`](../../state_machine/rl_control_state_onnx.hpp) | 采集机器人状态、周期驱动策略线程 | 负责控制频率、线程与安全检查 |
| 策略执行器 | [`run_policy/lite3_test_policy_runner_onnx.hpp`](../../run_policy/lite3_test_policy_runner_onnx.hpp) | 基于 ONNX Runtime 执行策略，将动作转换为关节命令 | RK3588 可替换为 Socket 驱动 |
| RKNN Socket Runner | [`run_policy/lite3_socket_policy_runner.hpp`](../../run_policy/lite3_socket_policy_runner.hpp) | 通过 Unix Socket 调用 `lite3_rknn_service.py` | 由环境变量 `LITE3_POLICY_BACKEND=RKNN_SOCKET` 启用 |
| 观测封装 | [`types/robot_basic_state.hpp`](../../types/robot_basic_state.hpp) *(若存在)* | 封装 IMU、关节、用户指令数据结构 | 供策略 Runner 使用 |
| 数据下发 | [`utils/data_streaming.hpp`](../../utils/data_streaming.hpp) | UDP 推送 PlotJuggler 数据、CSV 记录 | 保障调试与归档 |

### 4.1 `RLControlStateONNX::PolicyRunner` 核心逻辑
```cpp
while (start_flag_){
    if(state_run_cnt_%policy_ptr_->decimation_ == 0 && state_run_cnt_ != run_cnt_record){
        clock_gettime(CLOCK_MONOTONIC,&start_timestamp);
        auto ra = policy_ptr_->GetRobotAction(rbs_);
        MatXf res = ra.ConvertToMat();
        ri_ptr_->SetJointCommand(res);
        run_cnt_record = state_run_cnt_;
        clock_gettime(CLOCK_MONOTONIC,&end_timestamp);
        policy_cost_time_ = (end_timestamp.tv_sec-start_timestamp.tv_sec)*1e3
                            +(end_timestamp.tv_nsec-start_timestamp.tv_nsec)/1e6;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}
```

### 4.2 `Lite3TestPolicyRunnerONNX` 观测组装与推理
```cpp
Vec3f base_omgea = ro.base_omega * omega_scale_;
Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
Vec3f cmd_vel = ro.cmd_vel_normlized.cwiseProduct(max_cmd_vel_);
for (int i = 0; i < act_dim_; ++i){
    joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
    joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
}
current_obs_ << base_omgea, projected_gravity, cmd_vel,
                joint_pos_rl - dof_pos_default_policy,
                joint_vel_rl;
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info_, current_obs_.data(), obs_dim_, input_shape.data(), input_shape.size());
auto reaction = session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1,
                             output_names_.data(), 1);
action = Eigen::Map<VecXf>(reaction.front().GetTensorMutableData<float>(), act_dim_);
```

### 4.3 `Lite3SocketPolicyRunner` 与 RKNN 通信
```cpp
const uint32_t obs_len = static_cast<uint32_t>(obs_dim_);
if (!SendAll(socket_fd_, &obs_len, sizeof(obs_len)) ||
    !SendAll(socket_fd_, obs.data(), sizeof(float) * obs_dim_)) {
    CloseSocket();
    throw std::runtime_error("[RKNN SOCKET] 发送观测失败，连接已断开");
}

uint32_t act_len = 0;
if (!RecvAll(socket_fd_, &act_len, sizeof(act_len)) || act_len != act_dim_) {
    CloseSocket();
    throw std::runtime_error("[RKNN SOCKET] 推理服务返回的动作维度不匹配");
}
std::vector<float> action_buffer(act_dim_);
if (!RecvAll(socket_fd_, action_buffer.data(), sizeof(float) * act_dim_)) {
    CloseSocket();
    throw std::runtime_error("[RKNN SOCKET] 接收动作数据失败，连接已断开");
}
action_ = Eigen::Map<VecXf>(action_buffer.data(), act_dim_);
```

**RK3588 适配建议：**
- 设置 `LITE3_POLICY_BACKEND=RKNN_SOCKET` 即可切换到 RKNN Socket Runner，与 `lite3_rknn_service.py` 建立 Unix Socket 通道。
- `decimation_` 控制策略更新频率，需与控制周期（1 kHz）匹配，避免过采样。

---

## 5. 通信与数据链路

### 5.1 控制面：Unix Socket

```text
C++ 控制线程  ←→  lite3_rknn_service.py  ←→  RKNN Runtime  ←→  RK3588 NPU
```

- C++ 端负责采集观测、打包 `float32` 数组并发送。
- 服务端完成归一化、推理，返回动作数组。
- 若需要跨设备，可在服务端启用 `--tcp 0.0.0.0:5555` 切换为 TCP。

### 5.2 数据面：UDP + CSV

`utils/data_streaming.hpp` 提供双通道能力：
- **在线可视化**：UDP 推送 JSON 到 PlotJuggler，端口可配置。
- **离线分析**：自动生成 CSV，记录关节、IMU、指令与策略延迟。

### 5.3 单一来源资产管理

- 模型资产与归一化参数置于 `/opt/models/<version>/`，并在 `policy/rk3588/inputs/` 中维护同构结构。
- 通过 `systemd` 管理推理服务生命周期，确保无人系统 24/7 稳态运行。

---

## 6. 调试与排错索引

| 症状 | 排查方向 | 参考代码 |
| --- | --- | --- |
| RKNN 推理失败 | 检查 `convert_to_rknn.py` 配置与量化数据集 | `policy/rk3588/convert_to_rknn.py` |
| 控制线程阻塞 | 优化 `RLControlStateONNX::PolicyRunner` 中的线程睡眠与 decimation | `state_machine/rl_control_state_onnx.hpp` |
| 动作震荡 | 确认 `Lite3TestPolicyRunnerONNX` 的归一化尺度、默认站姿 | `run_policy/lite3_test_policy_runner_onnx.hpp` |
| 数据链路无输出 | 检查 UDP/TCP 端口、CSV 权限 | `utils/data_streaming.hpp` |

---

## 7. 对军民融合交付的意义

- **战略一致性**：该流水线将训练、量化、端侧推理到控制执行的每一环节固化为可审计代码资产，满足《院科字[2023]18号》单一来源协议直采的规范要求。
- **战术实时性**：通过 RK3588 NPU 与 Lite3 控制栈协同，实现无人系统亚 5 ms 的策略推理闭环，支撑高动态战场环境的自主决策。
- **持续演进**：模块化代码索引使得策略更迭、硬件升级或多平台扩展时，可快速定位改造点，保障大湾区军民融合创新生态的快速响应能力。

---


