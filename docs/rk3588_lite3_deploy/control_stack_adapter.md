# Lite3 控制栈 RKNN 适配技术说明

> 撰写人：中兵智能创新研究院(深圳)有限公司 技术总监  
> 项目：9G 大模型端侧部署专项（Lite3 Deploy / RK3588）

本说明书聚焦 Lite3 控制栈（C++）如何对接 RK3588 板端的 `policy.rknn` 推理服务，覆盖启动流程、代码入口、通信协议与运行环境约束。文档面向执行单一来源协议直采任务的工程团队，确保 `policy.pt` 经端侧固化后，可在 Lite3 实机实现 **毫秒级闭环控制** 与 **军民融合装备体系** 的长期运维。

---

## 1. 控制栈入口与状态机调度

- **启动命令**：
  ```bash
  ./lite3_deploy
  ```
  `main.cpp` 仅初始化状态机并进入循环，所有策略逻辑由 `StateMachine` 统一调度。【F:main.cpp†L1-L17】

- **状态机装配**：
  `StateMachine` 构造函数实例化了待机、站起、RL 控制、关节阻尼等状态，并在硬件模式下默认挂载 `HardwareInterface`、`RetroidGamepadInterface`、`DataStreaming` 等关键组件，用于接入 Lite3 机身总线与操控端。【F:state_machine/state_machine.hpp†L1-L152】

- **RL 控制入口**：
  `RLControlStateONNX` 是 RL 控制状态的主执行体，其 `OnEnter` 会启动独立线程 `PolicyRunner()`，在 1 kHz 控制节拍下拉取策略动作并同步给机器人接口。【F:state_machine/rl_control_state_onnx.hpp†L15-L149】

## 2. RKNN Socket 策略运行器

为保持 C++ 控制栈与 Python 推理服务解耦，我们新增 `Lite3SocketPolicyRunner`：

- **启用方式**：在部署环境设置
  ```bash
  export LITE3_POLICY_BACKEND=RKNN_SOCKET
  export LITE3_RKNN_SOCKET=/tmp/lite3_rknn.sock  # 若使用非默认路径
  ```
  `RLControlStateONNX` 会根据环境变量选择 Socket 后端，并输出实际使用的后端信息。【F:state_machine/rl_control_state_onnx.hpp†L90-L108】

- **核心职责**：
  - 建立 Unix Domain Socket 连接，默认路径 `/tmp/lite3_rknn.sock`。【F:run_policy/lite3_socket_policy_runner.hpp†L115-L138】【F:run_policy/lite3_socket_policy_runner.hpp†L234-L245】
  - 组装观测向量（IMU、关节状态、上一帧动作），并以 `<uint32长度, float32数组>` 格式发送给服务端。【F:run_policy/lite3_socket_policy_runner.hpp†L162-L180】【F:run_policy/lite3_socket_policy_runner.hpp†L247-L257】
  - 接收 RKNN 推理返回的动作向量，完成关节重排与比例缩放后转写为 `RobotAction`。【F:run_policy/lite3_socket_policy_runner.hpp†L259-L289】

- **异常处理**：若发送或接收失败，运行器会关闭 socket 并抛出异常，防止机器人保持过时命令，方便上层捕获并执行安全状态切换。【F:run_policy/lite3_socket_policy_runner.hpp†L253-L274】

## 3. 推理服务与环境要求

- **服务部署**：`lite3_rknn_service.py` 在 RK3588 板端加载 `policy.rknn` 与 `obs_stats.npz`，监听 Unix Socket，执行归一化与 NPU 推理。【F:policy/rk3588/lite3_rknn_service.py†L1-L86】

- **运行环境**：需预装 `rknn-toolkit2` 与 Rockchip 提供的 `librknnrt` runtime，同时保持 Python 环境与文档中 `requirements.txt` 对齐。服务可通过 systemd 常驻运行，保证任务期间 24/7 在线。【F:docs/rk3588_lite3_deploy/runtime_service.md†L7-L135】

- **通信协议**：C++ 控制栈与服务端使用相同的小端 `uint32` 头格式（4 字节长度 + `float32` payload），确保在本地/跨进程扩展时无需修改协议栈，实现动作与观测的高速无损传输。【F:policy/rk3588/lite3_rknn_service.py†L13-L78】【F:run_policy/lite3_socket_policy_runner.hpp†L82-L106】

## 4. 闭环执行流程

1. **策略导出**：按照部署流水线 (`export_pt_to_onnx.py` → `convert_to_rknn.py`) 产出 `policy.rknn` 与 `obs_stats.npz`，放置于 `/opt/models/<version>/`。【F:docs/rk3588_lite3_deploy/deployment_pipeline.md†L1-L133】
2. **推理服务启动**：RK3588 端执行 `python3 lite3_rknn_service.py --model ... --obs-stats ...`，服务监听 Socket。【F:policy/rk3588/lite3_rknn_service.py†L33-L78】
3. **控制栈编译运行**：在 Lite3 控制端设置环境变量后启动 `lite3_deploy`，`RLControlStateONNX` 自动切换至 RKNN Socket Runner。【F:state_machine/rl_control_state_onnx.hpp†L90-L123】
4. **动作下发**：`PolicyRunner()` 周期性调用 `GetRobotAction()` 并将结果转换为关节命令，通过硬件接口推送到机器人本体完成闭环。【F:state_machine/rl_control_state_onnx.hpp†L70-L88】【F:run_policy/lite3_socket_policy_runner.hpp†L281-L289】【F:interface/robot/hardware/hardware_interface.hpp†L1-L88】

## 5. 安全与运维建议

- **失控判据**：`RLControlStateONNX::LoseControlJudge()` 当检测到姿态超过阈值或操控端切换模式时，会跳转到关节阻尼状态，配合 Socket 异常抛出形成双重保险。【F:state_machine/rl_control_state_onnx.hpp†L133-L149】

- **日志与可视化**：`DataStreaming` 模块实时采集关节、IMU、动作等数据，可用于 PlotJuggler 在线监控和 CSV 归档，支撑单一来源交付的可审计要求。【F:state_machine/state_machine.hpp†L58-L103】

- **版本治理**：建议在 `/opt/models/<version>/` 建立哈希清单，记录 `policy.rknn`、`obs_stats.npz`、Socket 协议版本等信息，并与 systemd 服务配置绑定，满足《院科字[2023]18号》对模型资产可追溯性的要求。【F:docs/rk3588_lite3_deploy/runtime_service.md†L55-L135】

---

通过上述适配，Lite3 控制栈在不改动现有状态机框架的前提下，即可加载 RK3588 NPU 上运行的强化学习策略，实现 **训练侧 → 量化侧 → 推理服务 → 控制执行** 的全链路闭环，为大湾区军民融合装备的自主控制提供高可靠的工程底座。

