# Water Plant EnvHub 实施计划

## 目标与边界

短期目标：在本仓库新建根目录 `src/`，从零实现一个可通过 LeRobot EnvHub 加载的 Water Plant Gymnasium 环境，并发布为可复现的 Hub 环境。现有 Python 文件仅作行为和资产参考，不作为新实现依赖；MuJoCo XML、机器人网格、纹理等资产可在确认许可证和依赖关系后复用。

本阶段只支持单一 `water_plant` 任务。暂不迁移其他 DexJoCo 任务、teleoperation、数据采集/转换、训练流水线和 OpenPI；也不把“EnvHub 可加载”误当作“已注册为 LeRobot 官方 `--env.type=water_plant`”。官方 benchmark 文档收录另列后续工作。

## 目标目录

```text
env.py                         # EnvHub 入口：make_env(...)
pyproject.toml                 # 包元数据和 Python 依赖
src/water_plant/
  __init__.py
  env.py                       # Gymnasium 环境和任务生命周期
  simulation.py                # MuJoCo 模型、控制器和仿真步进
  assets/                      # 所需 XML、网格、纹理等资源
tests/
  test_env.py                  # API、空间、seed、成功/终止测试
README.md                      # Hub 使用说明、依赖和评估接口
```

保持模块边界简单：`env.py` 处理 Gymnasium 契约和任务语义；`simulation.py` 封装 MuJoCo 细节；根 `env.py` 只负责 EnvHub 导入和创建单/向量环境。资源路径从包位置解析，不依赖当前工作目录。EnvHub 的 `src/` 布局需由入口显式加入导入路径或通过安装包确保可导入。

## 分步实施

### 1. 冻结首版环境契约

先确定并写入 README：动作定义与范围、状态向量字段顺序、相机键名/分辨率、控制频率、episode 上限、奖励与成功条件。建议首版采用单臂末端位姿加 Allegro 手指目标的连续动作，并把动作范围正确映射到工作空间/关节范围；观测按 LeRobot 约定输出 `{"pixels": {"front": HWC uint8, "wrist": HWC uint8}, "agent_pos": float32 向量}`。固定相机键名和状态字段顺序，环境提供 `task`、`task_description` 和 `_max_episode_steps`。成功时 `info["is_success"]` 为真；reset 也返回该字段。成功使用 `terminated`，时间上限使用 `truncated`。

### 2. 建立可分发的项目骨架

新建上述 `src/` 包、根级 EnvHub 入口、`pyproject.toml` 和 `uv.lock`。以 Python 3.11 为基线，使用 `uv` 声明/解析 Python 依赖并锁定版本；开发环境通过 `uv sync` 建立，测试和检查通过 `uv run` 执行，不再为 Python 包维护 Conda 环境。LeRobot 集成测试依赖可放入可选 dependency group/extra，避免仿真核心被 LeRobot 依赖拖重。只纳入 Water Plant XML 的传递依赖资源，保持 XML include 和 mesh/texture 相对路径有效。检查并记录复用资产的许可证；MuJoCo 资产随包或 Hub 仓库版本化，不从仓库外的绝对路径读取。EGL/图形驱动属于操作系统依赖，需单独记录安装前提。

### 3. 从零实现仿真和任务

实现 MuJoCo 加载、Panda 控制、手指控制、喷壶触发器、观测和渲染。首先完成无渲染的 reset/step，再接入相机。随机数只通过 Gymnasium 的 `np_random` 管理，确保 `reset(seed=...)` 可复现；每个 episode 清理触发器、成功计数、模型随机化和渲染状态。将成功判定实现为清晰的独立逻辑，并区分成功、失败条件和时间截断。

### 4. 实现 EnvHub 入口

根 `env.py` 提供文档要求的 `make_env(n_envs=1, use_async_envs=False, cfg=None)`，返回 LeRobot 接受的 Gymnasium 环境结构。默认用同步环境；异步向量化时在 worker 内创建 MuJoCo renderer/EGL 上下文，避免父进程预先创建图形资源。入口不应依赖本仓库其他旧包。

### 5. 自动化验证

至少覆盖：Gymnasium reset/step 五元组、observation/action space 与实际数据匹配、图像形状和 dtype、seed 可复现、episode 上限和成功标记、close 可重复调用。以 `uv run pytest` 执行单测，以 `uv run` 运行 `make_env()` 本地加载 smoke test；确认单环境和 `n_envs > 1` 的创建/关闭行为。用已知动作轨迹或可控测试条件覆盖成功判定，不以零动作测试代替成功测试。EGL 图像测试作为需要图形驱动的独立测试标记。

### 6. 发布 EnvHub 版本

README 记录 `uv sync` 安装流程、系统图形前提、观测/动作 schema、任务语言描述和加载示例。EnvHub 调用方需使用兼容的 LeRobot 环境，并按锁文件安装仿真依赖；先从本地克隆目录运行，再推送 Hub。通过 `lerobot.envs.make_env("组织/仓库@commit", trust_remote_code=True)` 加载固定 commit，检查 reset、step、视频和关闭流程。远程代码需明确提示信任风险；后续更新用新 commit/tag 发布，不让评估默认跟随浮动的 `main`。

## 验收标准

- 新实现只依赖 `src/` 包和其声明/打包的资源，不 import 旧 DexJoCo Python 模块。
- EnvHub `make_env()` 可创建单任务环境，LeRobot 能 reset、step 并读取 `is_success`。
- 同 seed 的初始场景一致；观测、动作空间和 episode 结束语义有测试覆盖。
- `uv.lock` 能在 Python 3.11 下复现 Python 依赖；系统 EGL/图形驱动前提另有说明。
- Hub 固定版本可从干净环境按 README 中的 `uv` 步骤加载并渲染至少一个 episode。

## 后续阶段

EnvHub 版本稳定后，再以该接口为基础准备 LeRobot benchmark PR：添加 `water_plant` 的 `EnvConfig` 注册、依赖声明、官方文档页、目录链接和 CI 测试。此阶段不要求先把所有环境代码搬进 LeRobot。
