# LeRobot 接口

使用 `uv sync` 安装分发 `lerobot_env_dexjoco`。Python 包仍使用 `dexjoco` 和
`dexjoco_data_converter`，原有转换、回放命令不变。LeRobot 0.6.1 的官方 CLI
会自动导入 `lerobot_env_dexjoco`，注册 `--env.type=dexjoco`。

环境返回 `agent_pos`（23 维状态）和 `pixels`（`front`、`wrist` 两路 HWC uint8
图像）。官方预处理将它们转换为 `observation.state` 和
`observation.images.<camera>`。动作仍为 22 维：位置、旋转向量和 16 个手指目标。
`render_mode=none` 时不返回或声明图像。

已有数据集及项目的转换、重新录制工具继续使用原来的 features，包括
`observation.state`、`observation.images.<camera>` 和 float64 状态、动作。
环境兼容官方评测和视频保存，不支持官方的 `--eval.recording=true` 轨迹录制流程。
当前无需额外的环境 processor。

使用匹配这些状态、动作和相机约定的策略检查点进行评测：

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=dexjoco \
  --policy.path=/path/to/pretrained_model \
  --eval.batch_size=1 \
  --eval.n_episodes=5
```

官方评测默认保存视频，使用 `render_mode=rgb_array`（默认值）。数据集若使用
`random_camera` 等不同相机名，需要通过官方 `--rename_map` 映射到策略期望的名字。
视频及数据集帧率保持原来的 30 fps，仿真控制频率仍为 50 Hz。

# 测试流程

环境迁移步骤与 water plant 示例见 [迁移指南](docs/迁移指南.md)。

先下载原始数据，然后转换成lerobot数据放在./datasets里面

参考
```bash
uv run dexjoco-convert \
  --task water_plant \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/water_plant \
  --output-dir datasets/water_plant \
  --n-envs 16
```

然后运行测试，检查replay是否成功
```bash
uv run pytest
```

经过转换筛选之后，剩余的episode数量可能不足100

抗锯齿的bug：只能保证环境里的图片是确定的，但是数据集里的图片是有损压缩的
