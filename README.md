# LeRobot 接口

使用 `uv sync` 安装分发 `lerobot_env_dexjoco`。Python 包仍使用 `dexjoco` 和
`dexjoco_data_converter`，原有转换、回放命令不变。LeRobot 0.6.1 的官方 CLI
会自动导入 `lerobot_env_dexjoco`，注册 `--env.type=dexjoco`。

v1.1 原始数据位于 `/data/dexjoco_raw_datasets_v1.1`，新转换的 LeRobot 数据位于
`datasets/lerobot_v1.1/{rand_obj,rand_full}/<task>`。转换保留全部轨迹；每个数据集的
`replay_validation.json` 保存筛选前逐条回放结果。

例如，从 v1.1 转换 water_plant 并验证全部轨迹：

```bash
uv run dexjoco-convert \
  --task water_plant \
  --zarr-dataset-dir /data/dexjoco_raw_datasets_v1.1/dexjoco_raw_datasets/water_plant \
  --output-dir datasets/lerobot_v1.1/rand_obj/water_plant \
  --no-delete-failed-episodes \
  --n-envs 4
```

转换和回放入口会在导入 NumPy 前配置 Haswell 数值内核。若调用方已导入 NumPy，
应在启动 Python 前设置以下环境变量：

```bash
export OPENBLAS_CORETYPE=HASWELL
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
```

微波炉和拍照任务的原始 `state` 在第 61–92 列保存固定的
`initial_hand_joint`（右手 16 维、左手 16 维）。该字段转换后保存在
`replay_initial_values.jsonl`，不增加策略机器人观测的维数。

转换和 `dexjoco-record-replay` 同时保存 float64 `action`（原始四元数动作，单臂 23 维、
双臂 46 维）和 `action_rotvec`（由同一份四元数动作生成，单臂 22 维、双臂 44 维）。
默认环境、数据集回放和重新录制使用四元数。环境配置可通过
`--env.action_format=rotvec` 切换；数据集回放和重新录制使用 `--action-format rotvec`，
此时读取 `action_rotvec`。LeRobot 常规训练默认使用 `action`；训练旋转向量策略时，
需显式选择对应字段、配置动作维数，并使用旋转向量环境。

新数据集全部通过回放校验后，可运行
`uv run pytest tests --replay-dataset-root datasets/lerobot_v1.1` 测试此版本。
普通 `uv run pytest` 默认跳过完整数据集回放，仅在显式传入 `--replay-dataset-root` 时运行。

必须使用 `--env.task=water_plant`、`--env.task=fold_glasses`、`--env.task=click_mouse`、
`--env.task=pinch_tongs`、`--env.task=pick_bucket`、`--env.task=hammer_nail` 或
`--env.task=bimanual_hanoi`、`--env.task=bimanual_microwave_cook`、`--env.task=bimanual_assembly`、`--env.task=bimanual_unlock_ipad`、`--env.task=bimanual_photograph` 明确选择任务。
双臂任务同时设置 `--env.bimanual=true`；`bimanual` 独立决定状态、动作和腕部相机特征。
任务 ID 按论文任务概览图逐行排列，water_plant 为 0，fold_glasses 为 1，click_mouse 为 2，
pinch_tongs 为 3，pick_bucket 为 4，hammer_nail 为 5，bimanual_microwave_cook 为 6，bimanual_unlock_ipad 为 7，bimanual_hanoi 为 8，bimanual_assembly 为 9，bimanual_photograph 为 10。

| 任务 | agent_pos | 动作 | pixels 相机 |
| --- | --- | --- | --- |
| water_plant | 23 维 | 23 维（可选 22） | ego、ego_left、ego_right、wrist |
| fold_glasses | 23 维 | 23 维（可选 22） | ego、ego_left、ego_right、wrist |
| click_mouse | 23 维 | 23 维（可选 22） | ego、ego_left、ego_right、wrist |
| pinch_tongs | 23 维 | 23 维（可选 22） | ego、ego_left、ego_right、wrist |
| pick_bucket | 23 维 | 23 维（可选 22） | ego、ego_left、ego_right、wrist |
| hammer_nail | 23 维 | 23 维（可选 22） | ego、ego_left、ego_right、wrist |
| bimanual_microwave_cook | 46 维 | 46 维（可选 44） | ego、wrist_left、wrist_right |
| bimanual_unlock_ipad | 46 维 | 46 维（可选 44） | ego、wrist_left、wrist_right |
| bimanual_hanoi | 46 维 | 46 维（可选 44） | ego、wrist_left、wrist_right |
| bimanual_assembly | 46 维 | 46 维（可选 44） | ego、wrist_left、wrist_right |
| bimanual_photograph | 46 维 | 46 维（可选 44） | ego、wrist_left、wrist_right |

图像为 HWC uint8。官方预处理将观测转换为 `observation.state` 和
`observation.images.<camera>`。默认动作为位置、wxyz 四元数和手指目标。所有环境使用原 dm_robotics 姿态控制器。
双臂状态和两种动作格式均排列为右臂位姿、右手关节、左臂位姿、左手关节；
状态旋转为 wxyz 四元数。原始 raw 数据的状态仍为双臂位姿在前、双手关节在后，转换时重排。
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

Pinch Tongs 的两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task pinch_tongs \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/pinch_tongs \
  --output-dir datasets/rand_obj/pinch_tongs \
  --n-envs 4
uv run dexjoco-convert \
  --task pinch_tongs \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/pinch_tongs \
  --output-dir datasets/rand_full/pinch_tongs \
  --n-envs 4
```

回放和重新录制均使用 `--task pinch_tongs`。恢复信息单独保存夹钳位姿
`orig_tongs_pose` 和 `table_delta_height`。夹钳初始关节角保持为 0.3，
累计三次开合并在桌面上方至少 0.1 米持续保持 30 个控制步后成功。
原始 `rand_obj` 数据包含四个相机，`rand_full` 只包含主视角和腕部相机；
转换和重新录制均保留源数据集的相机集合。

2026-10-30 的完整数据回放结果如下，成功率均以筛选前的 100 条原始轨迹为分母：

| 数据组 | 迁移版 | conda dexjoco 参考版 |
| --- | --- | --- |
| rand_obj | 84/100（84%） | 86/100（86%） |
| rand_full | 85/100（85%） | 86/100（86%） |

参考验证调用 `dexjoco_ref/scripts/replay_demos_zarr.py`，使用
`--randomize=False --restore_state=True --extend_steps=60 --seed=0`。
迁移版同样恢复记录的初始场景，关闭视觉及动力学随机化，延长回放 60 步。
两边均为 MuJoCo 3.4.0；迁移环境使用 NumPy 2.2.6、SciPy 1.18.1，
参考环境使用 NumPy 1.26.4、SciPy 1.17.1。
筛选后保留的 169 条轨迹全部通过重复状态和图像一致性测试；
`rand_full` 的全部 85 条轨迹也完成视觉随机化重新录制，features、相机集合、
帧率及 float64 动作前缀均与源数据一致。

Click Mouse 的两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task click_mouse \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/click_mouse \
  --output-dir datasets/rand_obj/click_mouse \
  --n-envs 4
uv run dexjoco-convert \
  --task click_mouse \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/click_mouse \
  --output-dir datasets/rand_full/click_mouse \
  --n-envs 4
```

回放和重新录制使用 `--task click_mouse`。恢复信息单独保存 `orig_mouse_pose`
和 `table_delta_height`，鼠标垫位置由鼠标初始位置和桌高恢复，显示器按原环境固定布局恢复。
鼠标左键相对初始位置的位移大于 0.001，且鼠标位于鼠标垫内时，屏幕变蓝；
屏幕变蓝后鼠标在垫内连续保持 10 个控制步即成功，超时为 1200 个控制步。
`rand_obj` 的四个相机及 `rand_full` 的主视角、腕部相机均按源数据保留。

2026-10-30 的完整回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 回放成功 | 筛选后保留 |
| --- | --- | --- |
| rand_obj | 99/100（99%） | 99 |
| rand_full | 100/100（100%） | 100 |

两组均未触发低于 95% 时的参考回放验证规则。保留的 199 条轨迹全部通过
重复状态、图像一致性及 float64 动作读取精度测试。`rand_full` 全部 100 条轨迹
也完成视觉随机化重新录制；features、相机集合、帧率和初始状态记录保持一致，
float64 回放动作通过逐条精确比较。

Fold Glasses 的两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task fold_glasses \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/fold_glasses \
  --output-dir datasets/rand_obj/fold_glasses \
  --n-envs 4
uv run dexjoco-convert \
  --task fold_glasses \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/fold_glasses \
  --output-dir datasets/rand_full/fold_glasses \
  --n-envs 4
```

回放和重新录制使用 `--task fold_glasses`。恢复信息单独保存 `orig_glass_pose`、
`orig_box_pose` 和 `table_delta_height`。两侧镜腿角度均大于 1.1，且眼镜在盒内
连续保持 50 个控制步后成功；超时为 1200 个控制步。盒内判定保留原实现的
世界坐标轴边界，不随盒子偏航旋转。动力学随机化保留原参考代码的实际作用范围：
眼镜自由关节的 Y/Z 摩擦、两侧镜腿的刚度和眼镜主体质量。
原始 `rand_obj` 包含四个相机，`rand_full` 只包含主视角和腕部相机；
转换及重新录制均保留源数据集的相机集合。

2026-10-30 的完整数据回放结果如下，成功率均以筛选前的 100 条原始轨迹为分母：

| 数据组 | 迁移版 | conda dexjoco 参考版 |
| --- | --- | --- |
| rand_obj | 89/100（89%） | 92/100（92%） |
| rand_full | 91/100（91%） | 92/100（92%） |

参考验证直接调用 `dexjoco_ref/scripts/replay_demos_zarr.py` 的
`_replay_single_demo`，通过原环境已有的 `image_obs=False` 关闭逐帧图像观测。
两组均回放全部 100 条原始轨迹，恢复记录的初始场景，关闭视觉及动力学随机化，
延长最后动作 60 步；基准 seed 为 0，按排序后的轨迹使用 1–100 的环境 seed。
参考代码的动作处理、状态恢复、控制时间步、动力学和成功条件均保持原样。
筛选保留的 180 条轨迹全部通过重复状态、图像一致性及 float64 动作读取精度测试，
转换后的动作和状态也与对应原始 Zarr 逐值一致。`rand_full` 全部 91 条保留轨迹
完成视觉随机化重新录制，features、相机集合、帧率和初始状态记录保持一致，
float64 回放动作通过逐条精确比较。

Pick Bucket 的两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task pick_bucket \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/pick_bucket \
  --output-dir datasets/rand_obj/pick_bucket \
  --n-envs 4
uv run dexjoco-convert \
  --task pick_bucket \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/pick_bucket \
  --output-dir datasets/rand_full/pick_bucket \
  --n-envs 4
```

回放和重新录制使用 `--task pick_bucket`。恢复信息单独保存 `orig_bucket_pose`、
`orig_boxed_food_pose` 和 `table_delta_height`。恢复场景后，重新记录四个桶底参考点的
初始高度。食物参考位置位于桶的八个参考点形成的世界坐标轴包围盒内，且四个桶底
参考点均比各自初始高度至少高 0.15 米时立即成功；超时为 1200 个控制步。
动力学随机化保留桶和食物的独立质量扰动，以及把手摩擦沿当前值逐次 reset 累乘的
原始行为。`rand_obj` 的四个相机和 `rand_full` 的主视角、腕部相机均按源数据保留。

2026-10-30 的完整数据回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 迁移版 | conda dexjoco 参考版 |
| --- | --- | --- |
| rand_obj | 100/100（100%） | 100/100（100%） |
| rand_full | 100/100（100%） | 100/100（100%） |

参考验证直接调用 `dexjoco_ref/scripts/replay_demos_zarr.py` 的
`_replay_single_demo`，仅通过原环境已有的 `render_spec` 将参考图像设为 16×16。
两组均回放全部 100 条原始轨迹，恢复记录的初始场景，关闭视觉及动力学随机化，
延长最后动作 60 步；基准 seed 为 0，按排序后的轨迹使用 1–100 的环境 seed。
参考代码的动作处理、状态恢复、控制时间步、动力学和成功条件均保持原样。
迁移版两组全部 200 条轨迹均保留，转换后的 float64 动作、状态与对应原始 Zarr
逐值一致，全部通过重复状态、图像一致性及 float64 动作读取精度测试。
`rand_full` 全部 100 条轨迹完成视觉随机化重新录制，features、相机集合、帧率和
初始状态记录保持一致，float64 回放动作通过逐条精确比较。

Hammer Nail 的两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task hammer_nail \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/hammer_nail \
  --output-dir datasets/rand_obj/hammer_nail \
  --n-envs 4
uv run dexjoco-convert \
  --task hammer_nail \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/hammer_nail \
  --output-dir datasets/rand_full/hammer_nail \
  --n-envs 4
```

回放和重新录制使用 `--task hammer_nail`。恢复信息单独保存 `orig_hammer_pose`、
`orig_nail_pose` 和 `table_delta_height`，钉子高度按原始恢复脚本由桌高重新计算。
锤头接触钉子且近期竖直速度低于 −0.02 时，按速度倍率累计打入深度；每次基础增量
为 0.008 米，倍率上限为 3，深度上限为 0.0726 米。深度达到 0.04 米立即成功，
超时为 1000 个控制步。速度保留最近 12 个物理子步样本，并沿用原实现的 0.02 秒
控制时间步作为差分除数。动力学随机化只扰动锤子质量，范围为基准值的 0.75–1.25 倍。

2026-10-30 的完整回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 回放成功 | 筛选后保留 |
| --- | --- | --- |
| rand_obj | 99/100（99%） | 99 |
| rand_full | 100/100（100%） | 100 |

两组均未触发低于 95% 时的参考回放验证规则。保留的 199 条轨迹全部通过重复状态、
图像一致性及 float64 动作读取精度测试。转换数据的 float64
动作、状态与对应原始 Zarr 逐值一致，相机集合按源数据保留。`rand_full` 全部
100 条完成视觉随机化重新录制，features、相机集合、帧率、初始状态记录及 float64
回放动作均通过核对。

Bimanual Microwave Cook 使用 `--env.task=bimanual_microwave_cook --env.bimanual=true`。
两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task bimanual_microwave_cook \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/bimanual_microwave_cook \
  --output-dir datasets/rand_obj/bimanual_microwave_cook \
  --n-envs 4
uv run dexjoco-convert \
  --task bimanual_microwave_cook \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/bimanual_microwave_cook \
  --output-dir datasets/rand_full/bimanual_microwave_cook \
  --n-envs 4
```

回放和重新录制使用 `--task bimanual_microwave_cook`。恢复信息单独保存
`orig_hot_dog_pose`、`orig_microwave_pose` 和 `table_delta_height`。
食物参考点位于微波炉局部坐标系的内部包围盒内（边界向外放宽 0.01 米）、
门关节角绝对值小于 0.01，且启动按钮存在距离不大于零的接触时立即成功。
超时为 1100 个控制步。桌腿中心下降桌高增量的一半、半长增加同样数值；
动力学随机化独立扰动门摩擦、微波炉质量及食物质量，各为基准值的 0.75–1.25 倍。
微波炉 OBJ 的两条无效材质文件引用已删除，仍使用原 XML 指定的 `Oven` 材质和纹理。

2026-10-30 的完整数据回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 回放成功 | 筛选后保留 |
| --- | --- | --- |
| rand_obj | 97/100（97%） | 97 |
| rand_full | 100/100（100%） | 100 |

两组均未触发低于 95% 时的参考回放验证规则。保留的 197 条转换数据的 float64
动作、状态与对应原始 Zarr 逐值一致；三相机集合按源数据保留。全部 197 条轨迹
通过重复状态、图像一致性及 float64 动作读取精度测试。
`rand_full` 全部 100 条轨迹完成视觉随机化重新录制，features、相机集合、帧率、
初始状态记录及 float64 回放动作均通过核对。

Bimanual Assembly 使用 `--env.task=bimanual_assembly --env.bimanual=true`。
两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task bimanual_assembly \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/bimanual_assembly \
  --output-dir datasets/rand_obj/bimanual_assembly \
  --n-envs 4
uv run dexjoco-convert \
  --task bimanual_assembly \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/bimanual_assembly \
  --output-dir datasets/rand_full/bimanual_assembly \
  --n-envs 4
```

回放和重新录制使用 `--task bimanual_assembly`。恢复信息单独保存 `orig_socket_pose`、
`orig_peg_pose` 和 `table_delta_height`。两物体均为自由关节，分别恢复完整位姿。
插销的指定圆柱碰撞体与插座底部接触体连续接触 30 个控制步即成功；接触中断后
重新计数，超时为 1500 步。动力学随机化独立扰动插销及插座质量，范围为基准值的
0.75–1.25 倍。两组数据均保留主视角和左右腕部相机。

2026-10-31 的完整数据回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 迁移版 | conda dexjoco 参考版 |
| --- | --- | --- |
| rand_obj | 94/100（94%） | 95/100（95%） |
| rand_full | 94/100（94%） | 96/100（96%） |

两组均按低于 95% 的规则完成参考验证，调用原始
`dexjoco_ref/scripts/replay_demos_zarr.py` 的 `_replay_single_demo`，通过原环境已有的
`image_obs=False` 关闭逐帧图像观测。两组各回放全部 100 条原始轨迹，恢复记录的
初始场景，关闭视觉及动力学随机化，延长最后动作 60 步；基准 seed 为 0，按排序
后的轨迹使用 1–100 的环境 seed。原始控制时间步、动力学和成功条件保持原样。
保留的 188 条转换数据的 float64 动作、状态与对应原始 Zarr 逐值一致，全部通过
重复状态、图像一致性及 float64 动作读取精度测试。
`rand_full` 的全部 94 条保留轨迹完成视觉随机化重新录制，features、相机集合、
帧率、初始状态记录及 float64 回放动作均通过核对。

Bimanual Unlock iPad 使用 `--env.task=bimanual_unlock_ipad --env.bimanual=true`，任务 ID 为 7。
两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task bimanual_unlock_ipad \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/bimanual_unlock_ipad \
  --output-dir datasets/rand_obj/bimanual_unlock_ipad \
  --n-envs 4
uv run dexjoco-convert \
  --task bimanual_unlock_ipad \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/bimanual_unlock_ipad \
  --output-dir datasets/rand_full/bimanual_unlock_ipad \
  --n-envs 4
```

回放和重新录制使用 `--task bimanual_unlock_ipad`。恢复信息单独保存 `orig_stand_pose`、
`orig_ipad_pose` 和 `table_delta_height`，分别恢复支架、iPad 完整位姿和桌高。
动力学随机化只扰动 iPad 质量，范围为基准值的 0.75–1.25 倍。
主视角统一为 `ego`，左右腕部相机按源数据保留。

解锁使用 `ENTERING`、`ANIMATING`、`CONFIRMING`、`SUCCEEDED`、`FAILED` 五个明确状态。
只有右手指尖与按钮的新接触计为输入，持续按住不重复计数；同一步的新按键沿用
原几何体 ID 顺序，密码完成或输错后立即结束本步输入处理。输错即失败，本轮不允许重试。
默认密码为 123，关闭动画；直接构造 `BimanualUnlockIpadEnv` 时可传入 `password`
和 `ipad_screen_effect`，例如 `password=[1, 1, 0], ipad_screen_effect=True`。
非空密码允许重复数字及 0–9 中的任意数字。

关闭动画时，最后一位正确输入当步进入确认并计为第一次检测，累计 10 个控制步检测即成功。
开启动画时，保留原来的 12 次延迟更新和 18 次渐变更新，再确认成功；最后一位匹配当步
计为第一次动画更新，动画完成当步计为第一次确认。动画进度由控制步推进，与是否创建
renderer 无关。`get_unlock_status()` 只读取状态，不推进计数。
成功与输错通过 `terminated` 结束，1200 步超时按现有环境模板通过 `truncated` 结束，
同时记录 `failure_reason="timeout"`；所有结束状态保持不变，直到 reset。

2026-11-01 的完整数据回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 迁移版 | conda dexjoco 参考版 | 筛选后保留 |
| --- | --- | --- | --- |
| rand_obj | 58/100（58%） | 58/100（58%） | 58 |
| rand_full | 100/100（100%） | 未触发低于 95% 的验证规则 | 100 |

`rand_obj` 按低于 95% 的规则调用原始 `dexjoco_ref/scripts/replay_demos_zarr.py`
的 `_replay_single_demo`，在 `conda dexjoco` 环境回放全部 100 条原始轨迹。
通过原环境已有的 `image_obs=False` 关闭逐帧图像观测，恢复记录的初始场景，
使用默认密码 123、关闭屏幕动画及视觉/动力学随机化，延长最后动作 60 步。
基准 seed 为 0，按排序后的轨迹使用 1–100 的环境 seed；控制时间步、动力学和
成功条件保持原样。两边不仅成功率相同，成功的 58 条轨迹集合也完全一致。
参考环境为 MuJoCo 3.4.0、NumPy 1.26.4、SciPy 1.17.1。
保留的 158 条转换数据的动作、机器人状态及初始场景字段与对应原始 Zarr 逐值一致，
动作读取保持 float64 精度。

`rand_full` 的全部 100 条轨迹完成视觉随机化重新录制，features、相机集合、帧率、
初始场景记录、float64 动作及帧数均通过核对；三路相机的 13 个视频文件均可读取。

全部 158 条保留轨迹通过重复回放的状态及三相机图像逐像素一致性测试。

Bimanual Photograph 使用 `--env.task=bimanual_photograph --env.bimanual=true`，任务 ID 为 10。
两组完整数据转换方式如下：

```bash
uv run dexjoco-convert \
  --task bimanual_photograph \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/bimanual_photograph \
  --output-dir datasets/rand_obj/bimanual_photograph \
  --n-envs 4
uv run dexjoco-convert \
  --task bimanual_photograph \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets_rand_full/bimanual_photograph \
  --output-dir datasets/rand_full/bimanual_photograph \
  --n-envs 4
```

回放和重新录制使用 `--task bimanual_photograph`。恢复信息单独保存 `orig_camera_pose`、
`orig_logo_pose` 和 `table_delta_height`，恢复相机完整自由关节位姿、标志局部位姿及桌高。
目标区域随标志位置更新。动力学随机化只扰动相机主体质量，范围为基准值的 0.75–1.25 倍。
主视角统一为 `ego`，左右腕部相机按源数据保留。

相机观察点位于目标圆柱区域内（半径 0.16 m、半高 0.10 m）、相机轴与世界 +X 方向
夹角不超过 10°，且快门与相机自身子树以外的几何体接触时，该控制步即成功。
不要求连续保持，超时为 1000 个控制步。
直接构造 `BimanualPhotographEnv` 时可传入 `camera_screen_effect=True`，启用相机背屏的
实时取景、快门闪光和照片冻结；默认关闭，与原始回放脚本一致。
`render_mode="none"` 不创建图形资源。窗口同步沿用 water_plant 的延迟创建方式，
拍照任务额外的纹理更新和视角可视性处理放在独立方法中。

2026-11-01 的完整数据回放结果如下，分母均为筛选前的全部原始轨迹：

| 数据组 | 回放成功 | 筛选后保留 |
| --- | --- | --- |
| rand_obj | 99/100（99%） | 99 |
| rand_full | 99/100（99%） | 99 |

两组均未触发低于 95% 时的参考回放验证规则。保留的 198 条转换数据的动作、
机器人状态及初始场景字段与对应原始 Zarr 逐值一致，动作读取保持 float64 精度。

`rand_full` 的全部 99 条保留轨迹完成视觉随机化重新录制，features、三相机集合、
30 fps 帧率、初始场景记录、float64 动作及帧数均通过核对；10 个视频文件均可读取。

全部 198 条保留轨迹通过重复回放的状态、float64 动作读取精度及三相机图像逐像素一致性测试。

经过转换筛选之后，剩余的episode数量可能不足100

抗锯齿的bug：只能保证环境里的图片是确定的，但是数据集里的图片是有损压缩的
