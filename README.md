# LeRobot 接口

使用 `uv sync` 安装分发 `lerobot_env_dexjoco`。Python 包仍使用 `dexjoco` 和
`dexjoco_data_converter`，原有转换、回放命令不变。LeRobot 0.6.1 的官方 CLI
会自动导入 `lerobot_env_dexjoco`，注册 `--env.type=dexjoco`。

必须使用 `--env.task=water_plant` 或 `--env.task=bimanual_hanoi` 明确选择任务。
双臂任务同时设置 `--env.bimanual=true`；`bimanual` 独立决定状态、动作和腕部相机特征。
任务 ID 按论文任务概览图逐行排列，water_plant 为 0，bimanual_hanoi 为 8。

| 任务 | agent_pos | 动作 | pixels 相机 |
| --- | --- | --- | --- |
| water_plant | 23 维 | 22 维 | ego、wrist |
| bimanual_hanoi | 46 维 | 44 维 | ego、wrist_left、wrist_right |

图像为 HWC uint8。官方预处理将观测转换为 `observation.state` 和
`observation.images.<camera>`。动作为位置、旋转向量和手指目标。
Hanoi 动作排列为右臂位姿、右手关节、左臂位姿、左手关节；状态排列为右臂位姿、
左臂位姿、右手关节、左手关节，旋转为 wxyz 四元数。
`render_mode=none` 时不返回或声明图像。

环境、已有数据集和新输出数据集的主视角统一为 `ego`，不随单双臂或随机化改变。
转换器将原始数据的 `front`、`random_camera` 映射为 `ego`，其他侧视角和腕部相机保留原名。
项目的转换、重新录制工具使用以下 features，包括
`observation.state`、`observation.images.<camera>` 和 float64 状态、动作。
环境兼容官方评测和视频保存，不支持官方的 `--eval.recording=true` 轨迹录制流程。
当前无需额外的环境 processor。

使用匹配这些状态、动作和相机约定的策略检查点进行评测：

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=dexjoco \
  --env.task=bimanual_hanoi \
  --env.bimanual=true \
  --policy.path=/path/to/pretrained_model \
  --eval.batch_size=1 \
  --eval.n_episodes=5
```

官方评测默认保存视频，使用 `render_mode=rgb_array`（默认值）。旧策略检查点若使用
不同主视角名称，需要通过官方 `--rename_map` 映射到策略期望的名字。
视频及数据集帧率保持原来的 30 fps，仿真控制频率仍为 50 Hz。

# 测试流程

环境迁移步骤与 water plant 示例见 [迁移指南](docs/迁移指南.md)。

静态检查使用以下命令。Pyright 与 Pylance 共用类型检查核心，项目配置会扫描源码、测试和入口文件，排除 `dexjoco_ref/` 和生成目录。

```bash
uv run ruff check
uv run mypy
uv run pyright
```

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

Hanoi 的两组完整数据转换方式如下：

```bash
MUJOCO_GL=egl uv run dexjoco-convert \
  --task bimanual_hanoi \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/bimanual_hanoi \
  --output-dir datasets/rand_obj/bimanual_hanoi
MUJOCO_GL=egl uv run dexjoco-convert \
  --task bimanual_hanoi \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/bimanual_hanoi \
  --output-dir datasets/rand_full/bimanual_hanoi
```

回放命令 `uv run python -m dexjoco.replay_dataset` 和重新录制命令
`uv run dexjoco-record-replay` 均使用 `--task bimanual_hanoi` 选择环境。
重新录制使用 `observation.images.ego` 主视角和对应任务的腕部相机。
Hanoi 的恢复信息单独保存底座位置 `orig_hanoi_base_pos` 和 `table_delta_height`；
盘片按原始环境的最后两步局面恢复：小盘在 A，中盘在 B，大盘在 C。

经过转换筛选之后，剩余的episode数量可能不足100

抗锯齿的bug：只能保证环境里的图片是确定的，但是数据集里的图片是有损压缩的
