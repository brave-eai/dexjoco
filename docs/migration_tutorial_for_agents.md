# Agent 环境迁移指南

## 工作原则

- 有任何不明确、有疑问或无法确认的地方，立即停止并向用户提问。用户愿意回答，不要担心打扰用户，不要猜测后继续实现。
- 遇到错误可以直接向用户求助。说明当前操作、错误和已有证据，不要盲目尝试各种修改。
- 一切代码以当前 `water_plant` 为参考。将它作为模板，填入当前任务对应的实现；环境、资产、配置、数据转换、回放和测试均沿用这个范例的组织方式。
- 代码风格以 `water_plant` 为准。不要使用 AI 式的超级防御性代码风格，不添加无实际需要的检查、兼容分支、异常吞噬、回退逻辑或复杂抽象。
- 公共流程通过环境属性或调用方配置表达差异，不按具体任务名硬编码行为。例如，单双臂由 `bimanual` 属性声明；回放并行数 `n_envs` 属于资源配置，由调用方设置，不与任务名或单双臂绑定。
- 所有代码注释和 docstring 使用英文。
- 不允许出现“契约”这个词。本文件中这条禁用说明是唯一例外，其他文件和新增内容不得使用。
- 使用 uv 管理依赖和运行命令。遵守仓库的 `AGENTS.md`。
- 不创建无关文件，清理自己产生的临时文件。如果需要临时文件记录迁移计划，文件名使用 `plan`。
- 修改前检查工作区，保留用户已有的修改。

## 阅读参考代码

先阅读 [现有迁移指南](../docs/迁移指南.md)，再阅读以下实现。所有路径相对于仓库根目录。

| 参考文件 | 阅读内容 |
| --- | --- |
| `src/dexjoco/envs/water_plant.py` | 环境实现模板，包括初始化、状态机、reset、step、观测、随机化、渲染和 close |
| `src/dexjoco/assets/tasks/water_plant/` | 场景 XML、物体及资产引用方式 |
| `src/dexjoco/assets/README.md` | 公共资产、机器人模型和任务资产的路径规则 |
| `src/dexjoco/controllers.py`、`src/dexjoco/common.py` | 当前环境使用的控制器和公共定义 |
| `src/dexjoco/configs.py`、`src/dexjoco/envhub.py` | 配置、特征声明和环境创建方式 |
| `src/lerobot_env_dexjoco/__init__.py`、根目录 `env.py` | 官方 CLI 注册入口及 EnvHub 加载入口 |
| `src/dexjoco_data_converter/` | water_plant 的数据读取、转换预设和转换后回放筛选流程 |
| `src/dexjoco/replay_dataset.py`、`src/dexjoco/record_replay.py` | 数据集回放和重新录制方式 |
| `tests/test_replay.py`、`tests/test_lerobot_interface.py` | 完整数据集验证和官方接口测试 |

原始参考仓库已迁移到 `dexjoco_ref/`，来源和版本见 [reference_upstream.txt](reference_upstream.txt)。随后阅读待迁移任务在 `dexjoco_ref/dexjoco/dexjoco/sim/envs/` 中的原始环境、对应 XML、`dexjoco_ref/dexjoco/dexjoco/tasks/<task>/config.py` 以及原始状态恢复和动作处理代码，查找需要填入模板的任务逻辑和参数。原始数据回放入口为 `dexjoco_ref/scripts/replay_demos_zarr.py`。

任务与 ID 的排列顺序参考仓库中的 [DexJoCo.pdf](../DexJoCo.pdf)。如果论文中的顺序、当前注册方式或任务名称不能明确对应，停止并询问用户。

原始代码用于查找任务行为，新代码的结构和风格以当前 `water_plant` 为准。遇到资料不一致或无法确定应保留哪种行为时，向用户确认。

## 迁移资产与 XML

参照 `water_plant` 的资产布局：

- 公共资源放入 `src/dexjoco/assets/common/`。
- 机器人模型放入 `src/dexjoco/assets/robots/`。
- 当前任务的场景和物体放入 `src/dexjoco/assets/tasks/<task>/`。
- 保留复用资产的许可证文件。

沿用现有 XML 规则：

- `include` 路径相对于包含它的 XML 文件。
- mesh、texture 文件路径统一相对于 `assets/` 根目录。
- `compiler` 只在场景入口 XML 中设置，被 include 的 XML 不重复配置。
- 场景设置 `<quality offsamples="0" />`，用于重复回放的图像一致性检查。
- 更新移动资产后的引用，确保加载不依赖工作目录或机器专属的绝对路径。

迁移过程中保留任务原有的控制和动力学参数。需要改变这些参数时，先向用户确认。

## 按 water_plant 模板实现环境

在 `src/dexjoco/envs/<task>.py` 中参照 `water_plant.py` 实现环境。保留相同的代码组织方式，将任务相关模型、名称、初始化数据、控制逻辑及观测填入对应位置。

### 初始化与资源管理

- 按名称查找当前任务使用的关节、执行器、传感器、物体和相机索引。
- 双臂的左右关节、执行器名称在 `src/dexjoco/common.py` 中独立定义，不复用单臂名称再重排或补后缀。
- 参照 `water_plant` 保存 reset 和随机化需要的基准值。
- 构造函数不创建 renderer 或 GUI，首次使用时再创建。
- 实现 `rgb_array`、`human` 和 `none` 模式，行为参考 `water_plant`。
- `close()` 释放当前环境拥有的渲染和窗口资源。

### reset 与初始状态恢复

- 通过 `reset(options={"initial_state": ...})` 恢复场景。
- 通过 reset 返回的 `info["initial_state"]` 提供恢复信息。
- 机器人状态中只包含机器人观测，物体位姿、桌高等恢复信息单独保存。
- 对照当前任务初始化逻辑，填写恢复所需的字段；不能确定字段是否足够时，向用户提问。
- options 只读取当前环境使用的字段，允许调用方携带其他选项，不对 options 设置排他性的断言。
- seed、随机化、机器人初始姿态和派生数据更新的实现方式参考 `water_plant`。

### 控制与任务状态机

- 按当前任务的动作表示实现控制，观测和动作的排列顺序与对应数据保持一致。
- 控制时间步、物理时间步和数据帧率沿用现有任务约定，不在迁移时顺带调整。
- 参照 `WaterPlantStateMachine` 提取任务的成功、失败和超时逻辑。
- reset 时重置任务计数器和状态。
- `step()` 返回 `(observation, reward, terminated, truncated, info)`，使用 `info["is_success"]` 报告成功。

### 观测与官方接口

- 机器人状态通过 `agent_pos` 返回。
- 图像通过 `pixels` 相机字典返回，图像格式为 HWC、uint8。
- 主视角统一命名为 `ego`，不随单双臂或视觉随机化改变；其他侧视角和腕部相机名称保持原样。
- `observation_space` 和 `action_space` 与实际返回及接收的数据一致。
- `render_mode="none"` 时不返回图像，也不创建图形资源。
- 提供 `task`、`task_description` 和 `_max_episode_steps`。
- `render()` 在 `rgb_array` 模式返回 RGB 数组，`metadata` 提供相应渲染模式和帧率。

本项目使用官方评测与视频保存，不接入官方轨迹录制。当前 `water_plant` 使用默认空环境 processor；新任务也以此为参考。出现额外观测或动作转换需求时，先说明需求并向用户确认，再决定是否添加 processor。

## 接入配置、转换和回放

所有接入方式参考 `water_plant`，逐项检查当前任务能否沿完整调用路径被选择和执行。

### 任务注册与配置

- 在 `src/dexjoco/envhub.py` 注册任务，任务与 ID 的排列参考论文。
- 配置中的 `task` 必须由调用方明确指定，不设置默认任务名；环境创建、转换、回放和重新录制入口遵循同一规则。
- `DexJoCoEnvConfig.bimanual` 是独立属性，用于声明单双臂对应的状态、动作和腕部相机特征，不通过某个任务名判断单双臂。
- 在配置中支持选择当前任务，并向构造函数传递对应参数。
- 按任务的状态、动作和相机声明 `features`、`features_map`。
- 图像特征尺寸跟随配置，`render_mode="none"` 时不声明图像特征。
- `features_map` 用于策略特征名称映射，实际观测仍须符合官方预处理的输入格式。
- 分发名保持 `lerobot_env_dexjoco`，由同名入口包导入配置，供官方 CLI 自动注册；内部代码沿用 `dexjoco` 包。
- 新资产和代码须包含在构建产物中。

### 数据转换、回放和重新录制

- 参考 water_plant 的转换预设，接入当前任务的动作、状态、任务描述及初始场景字段。
- 数据集继续使用 `action`、`observation.state`、`observation.images.<camera>` 等现有 features，维数和相机按当前任务填写。
- 环境和所有已有、新生成的数据集主视角均使用 `ego` 和 `observation.images.ego`。转换原始数据时，将 `front`、`random_camera` 主视角映射为 `ego`；其他相机名称保持原样。
- 除上述主视角名称统一外，保持既定数据集 schema、数值类型和维数。更新已有数据时可重新生成转换数据集，不删除原始数据。
- 恢复信息继续单独保存为 `replay_initial_values.jsonl`。
- 任务选择应贯穿环境创建、转换后回放筛选、数据集回放、重新录制和测试，不能仍固定使用 water_plant 或任务 ID 0。
- 正常的数据筛选沿用 water_plant 现有流程。
- 回放读取须保留数据集存储的动作精度。读取 float64 动作时关闭 LeRobot 默认的 float32 tensor transform，避免先量化再转回 float64，改变接触动力学和筛选结果。

## 验证与错误处理

直接使用当前任务的完整数据集完成转换与回放验证，不增加小样本验证阶段。数据位置、数据版本或运行条件不明确时，先向用户询问。

测试当前环境功能和实际接口，不为已删除的代码或过去的 bug 保留专门的兼容测试。测试代码同样参考 water_plant 的现有测试组织方式。

NumPy、SciPy 和底层 BLAS 的版本或构建不同，以及四元数与旋转向量转换中的细微浮点误差，都可能改变接触仿真的轨迹和成功率。由这类细微数值差异引起的成功率变化是正常且可以容忍的，不要求迁移版与原版的成功率或逐条成败完全相同，也不为对齐结果调整动力学参数或成功条件。

检查以下行为：

- 当前任务可以通过环境注册入口创建，并能同步、异步运行。
- 官方预处理得到的状态和图像与配置的特征名称、类型、形状相符。
- 官方 rollout 能执行 reset、动作和结束判断，RGB 视频可以保存和读取。
- 完整数据集能够恢复初始场景并回放，检查任务成功以及重复回放的状态、图像一致性。
- 图像一致性比较使用相同渲染条件下的重复回放，不要求与有损压缩视频逐像素一致。
- 转换和重新录制保持既定数据集 features。
- 环境关闭后释放其资源。

如果任一组完整数据的迁移版回放成功率低于 95%，必须在 `conda dexjoco`
环境中调用 `dexjoco_ref/scripts/replay_demos_zarr.py`，回放对应的完整原始数据，
核对参考代码的成功率是否也大约同样低。比较时记录两边的原始轨迹数、成功数、
成功率和回放设置；迁移版使用转换后筛选前的成功率，不能用筛选后的成功率比较。
可以临时删除参考仿真环境中的 `sleep` 调用以加速验证，不改变控制时间步、
动力学参数或成功条件；验证结束后恢复这些临时改动并清理临时输出。
如果两者成功率存在无法解释的明显差异，停止并向用户提供结果，不以筛选失败轨迹代替调查。

### GPU 渲染与沙箱

无窗口渲染使用 `MUJOCO_GL=egl`，例如：

```bash
MUJOCO_GL=egl uv run pytest tests/test_replay.py --replay-n-envs=4
```

EGL 启动成功不代表使用了 GPU。本机普通沙箱执行上述渲染时会使用 Mesa 的 `llvmpipe` 软件渲染；需要 GPU 时，通过执行工具的 `sandbox_permissions="require_escalated"` 在沙箱外运行命令，才能访问本机 NVIDIA GPU。使用 `GL_VENDOR`、`GL_RENDERER` 确认实际渲染后端。当前实现中 GPU 用于渲染，MuJoCo 物理模拟仍在 CPU 上运行。

如果受执行权限限制而无法运行，直接把完整命令交给用户运行。

### 数据转换与 Hugging Face 缓存权限

当前沙箱中，`uv run dexjoco-convert` 可以向仓库内的 `datasets/` 写入转换结果，
但转换后的回放筛选会通过 LeRobot 读取数据集。Hugging Face Datasets 在读取
Parquet 数据及元数据时也会向默认的 `~/.cache/huggingface/datasets/` 创建缓存锁；
该目录在普通沙箱中只读，因此可能在全部轨迹转换完成后报
`OSError: [Errno 30] Read-only file system`，路径以缓存目录中的 `.lock` 结尾。

使用默认缓存路径执行完整转换及回放筛选时，通过执行工具的
`sandbox_permissions="require_escalated"` 在沙箱外运行，例如：

```bash
uv run dexjoco-convert \
  --task pinch_tongs \
  --zarr-dataset-dir /data/dexjoco_raw_datasets/dexjoco_raw_datasets/pinch_tongs \
  --output-dir datasets/rand_obj/pinch_tongs \
  --n-envs 4
```

这里的额外权限用于访问用户缓存，不是写入转换数据或物理回放本身的要求。
无需 GPU 的转换和成功率筛选，也可将 `HF_DATASETS_CACHE` 指向沙箱内可写目录后运行；
`uv` 自身的默认缓存目录只读时，同样用 `UV_CACHE_DIR` 指向可写目录。
临时缓存使用完毕后清理。需要 NVIDIA 图像渲染时仍按上一节在沙箱外运行。

如果转换已完成，仅在后续读取缓存锁时报错，保留转换结果和
`replay_initial_values.jsonl`，修正执行权限或缓存位置后调用现有的
`replay_and_filter_dataset` 继续筛选，不必重复解码和转换全部原始视频。

### 检查命令

使用 uv 执行检查：

```bash
uv run ruff check
uv run mypy
uv run pyright
uv run pytest
```

Pyright 与 Pylance 共用类型检查核心，用于批量检测编辑器中的类型错误；检查范围、Python 版本和虚拟环境在 `pyproject.toml` 的 `[tool.pyright]` 中配置，排除 `dexjoco_ref/` 和生成目录。

数据集回放测试通过 `--replay-n-envs` 设置并行数，所有任务共用此配置，默认值为 4。例如：`uv run pytest tests/test_replay.py --replay-n-envs=4`。

遵循 `AGENTS.md` 中对暂存区测试的要求；提交前，确保测试覆盖最终准备提交的代码。

不要通过提高容差、放宽成功条件或删除失败轨迹来掩盖迁移错误。出现失败、异常或解释不清的差异时，可以立即停止并向用户求助，不要盲目尝试。需要用户判断时，提供操作、错误输出和已经确认的事实。

## 提交要求

只有用户明确要求时才创建 Git 提交。由 Codex 自动迁移的环境，在提交说明中明确注明该环境由 Codex 辅助生成。

提交前清理自己生成的临时文件，包括不再需要的 `plan`。
