# 推理服务与控制栈对接

## 1. 原则

- **解耦策略与控制**：以 Unix Socket/TCP 服务形式承载 RKNN 推理，保持 C++ 控制栈纯粹。

## 2. `lite3_rknn_service.py`

### 2.1 启动命令

```bash
python3 lite3_rknn_service.py \
    --model /opt/models/lite3_ppo_v5/policy.rknn \
    --obs-stats /opt/models/lite3_ppo_v5/obs_stats.npz \
    --socket /tmp/lite3_policy.sock \
    --warmup 32 \
    --log-level INFO
```

- `--tcp 0.0.0.0:5555` 可切换为 TCP 服务，方便跨设备通信。
- `--obs-dim`、`--act-dim` 可覆盖自动推断值，适用于策略结构变化场景。

### 2.2 核心流程

```python
class RK3588LocomotionService:
    def __init__(self, args):
        self.rknn = RKNN()
        self.rknn.load_rknn(args.model)
        self.rknn.init_runtime(target='rk3588')
        stats = np.load(args.obs_stats)
        self.obs_mean = stats['mean']
        self.obs_std = np.maximum(stats['std'], 1e-3)

    def _preprocess(self, obs):
        return ((obs - self.obs_mean) / self.obs_std).astype(np.float32)

    def infer(self, obs):
        norm = self._preprocess(obs)
        outputs = self.rknn.inference(inputs=[norm])
        return outputs[0].astype(np.float32)
```

Socket 读写：

```python
payload = client.recv(self.obs_dim * 4)
obs = np.frombuffer(payload, dtype=np.float32)
action = self.infer(obs)
client.sendall(action.tobytes())
```

### 2.3 systemd 管理示例

`/etc/systemd/system/lite3-rknn.service`：

```
[Unit]
Description=Lite3 RKNN Policy Service
After=network.target

[Service]
Type=simple
User=lite3
ExecStart=/home/lite3/envs/rk3588/bin/python /opt/Lite3_rl_deploy/policy/rk3588/lite3_rknn_service.py \
    --model /opt/models/lite3_ppo_v5/policy.rknn \
    --obs-stats /opt/models/lite3_ppo_v5/obs_stats.npz \
    --socket /tmp/lite3_policy.sock \
    --warmup 32
Restart=always
RestartSec=2
RuntimeDirectory=lite3_rknn
RuntimeDirectoryMode=0750

[Install]
WantedBy=multi-user.target
```

启停命令：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now lite3-rknn.service
sudo systemctl status lite3-rknn.service
```

## 3. C++ 控制栈接入

### 3.1 Socket 客户端骨架

```cpp
int sock = ::socket(AF_UNIX, SOCK_STREAM, 0);
struct sockaddr_un addr{};
addr.sun_family = AF_UNIX;
std::strncpy(addr.sun_path, "/tmp/lite3_policy.sock", sizeof(addr.sun_path));
::connect(sock, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr));

std::array<float, OBS_DIM> obs = CollectObservation();
::send(sock, obs.data(), obs.size() * sizeof(float), MSG_NOSIGNAL);
std::array<float, ACT_DIM> action;
::recv(sock, action.data(), action.size() * sizeof(float), MSG_WAITALL);
ApplyAction(action);
```

- 建议启用非阻塞 I/O，并配置 1~2 ms 的 `SO_RCVTIMEO` 防止超时。
- 若 C++ 直接集成 librknnrt，可参考服务内部逻辑转写。

### 3.2 与状态机集成

1. 在 `run_policy/policy_runner.cpp` 中新增 `SocketPolicyRunner`，在 `Tick()` 中执行上述通信。
2. 通过参数文件或命令行切换策略来源（本地 RKNN / 远程服务 / ONNX Runtime）。
3. 使用环形缓冲对观测进行时间对齐，避免控制线程阻塞。

## 4. 性能优化建议

- **预分配内存**：在 C++ 端将观测与动作缓冲置于共享内存，避免重复分配。
- **批量推理**：若一次性控制多条腿或多台机器人，可将 batch size 提升至 4，前提是保证 NPU 内存足够。
- **多线程调度**：将服务线程绑定至 big-core（`taskset -c 4-7`），控制线程绑定至实时核心（`chrt -f 90`）。
- **链路监控**：结合 `perf`、`tegrastats` 类工具监控 NPU 利用率和延迟波动。

## 5. 故障排查

| 症状 | 原因 | 处理措施 |
| --- | --- | --- |
| 服务启动即退出 | `policy.rknn` 路径错误或模型不兼容 | 检查路径，使用 `rknn.query()` 查看模型信息 |
| 推理耗时突然升高 | NPU 频率下降或线程被抢占 | 增加预热次数，使用 `cpupower frequency-set` 锁定频率 |
| C++ 收不到动作 | Socket 被关闭 | 检查服务日志，确认客户端异常断开后重新连接 |
| 动作异常震荡 | 归一化参数不匹配 | 确保 `obs_stats.npz` 与策略训练时一致 |

## 6. 灰度发布流程

1. 在仿真环境启动服务，使用 `run_rknn_agent.py` 验证动作序列。
2. 在备用机器人上验证实机表现，记录 IMU 与关节力矩曲线。
3. 将 systemd 单元部署至生产机器人，开启实时监控。
4. 建立回滚预案：保留旧版本 `.rknn` 与 systemd 配置，必要时一键切换。

