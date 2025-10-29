# 部署流水线细解：`.pt → .onnx → .rknn`

## 1. 准备

1. 导出 `policy.pt` 与 `obs_stats.npz`。
2. 确认 YAML 配置（如 `example_actor_config.yaml`）与模型结构保持一致：
   ```yaml
   obs_dim: 108
   act_dim: 12
   hidden_sizes: [512, 256]
   activation: tanh
   ```
3. 将资产置于 `policy/rk3588/inputs/` 目录以便脚本统一引用。

## 2. `.pt` 导出为 `.onnx`

```bash
python3 export_pt_to_onnx.py \
    ./inputs/policy.pt \
    --config example_actor_config.yaml \
    --output build/policy.onnx \
    --dynamic-batch \
    --opset 11
```

脚本关键逻辑：

```python
model = ActorCritic(obs_dim=cfg["obs_dim"], act_dim=cfg["act_dim"], hidden_sizes=cfg["hidden_sizes"], activation=cfg.get("activation", "tanh"))
state_dict = torch.load(args.pt, map_location="cpu")
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.zeros(1, cfg["obs_dim"], dtype=torch.float32)
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=["obs"],
    output_names=["action"],
    opset_version=args.opset,
    dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}} if args.dynamic_batch else None,
)
```

- 若 `policy.pt` 包含优化器状态，可先在训练侧保存纯权重；脚本对 `{"state_dict": ...}` 结构亦能自动处理。
- 建议 `opset >= 11` 以获得更佳算子支持。

## 3. 校准数据采集

```bash
python3 collect_calib_data.py \
    --env interface.robot.simulation.pybullet_env:make_env \
    --policy policy.rk3588.example_dummy_policy:RandomPolicy \
    --samples 2048 \
    --output build/calib_obs.npy \
    --batch 32
```

内部机制：

```python
env_factory = locate_callable(args.env)
policy_factory = locate_callable(args.policy)
env = env_factory()
policy = policy_factory(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0])
obs = env.reset()
for _ in tqdm(range(args.samples)):
    buffer.append(obs.astype(np.float32))
    action = policy.act(obs)
    obs, *_ = env.step(action)
    if done:
        obs = env.reset()
np.save(args.output, np.stack(buffer))
```

- `--batch` 参数控制一次写入的样本数量，便于应对超大样本量。
- 对于实机数据，可将 `env` 与 `policy` 替换为读取日志或在线采集脚本。

## 4. ONNX → RKNN

```bash
python3 convert_to_rknn.py build/policy.onnx \
    --dataset build/calib_obs.npy \
    --output build/policy.rknn \
    --target rk3588 \
    --do-quant \
    --quant-dtype asymmetric_affine-u8 \
    --optimization-level 3 \
    --output-optimize-level 1
```

核心实现：

```python
rknn = RKNN(verbose=args.verbose)
rknn.config(
    target_platform=args.target,
    optimization_level=args.optimization_level,
    output_optimize_level=args.output_optimize_level,
    quantized_dtype=args.quant_dtype,
    mean_values=None,
    std_values=None,
)
rknn.load_onnx(model=args.onnx)
if args.do_quant:
    rknn.build(do_quantization=True, dataset=args.dataset)
else:
    rknn.build(do_quantization=False)
rknn.export_rknn(args.output)
```

常见排错：

| 症状 | 原因 | 解决方案 |
| --- | --- | --- |
| `Load onnx failed` | ONNX 不兼容 | 使用 `onnxsim` 简化图，或降低 opset | 
| `Build failed` | 量化校准不足 | 增加样本数或改为浮点模型 | 
| `Invalid tensor dims` | 动态维度未设置 | 启用 `--dynamic-batch` 并在控制栈中固定 batch=1 |

## 5. 模型版本固化

- 将生成的 `policy.rknn` 及 `obs_stats.npz`、`policy.onnx`、`calib_obs.npy` 一并归档。
- 使用 Git LFS 或制式文件服务器存储，记录 `model_version.json`：
  ```json
  {
    "model": "lite3_ppo_v5",
    "onnx_hash": "sha256:...",
    "rknn_hash": "sha256:...",
    "quant_dataset": "calib_obs.npy",
    "export_time": "2024-05-12T08:00:00Z"
  }
  ```
- 在端侧 `/opt/models/<version>/` 建立独立目录。

