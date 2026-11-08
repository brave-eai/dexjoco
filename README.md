<p align="center">
  <img src="docs/pics/dexjoco_logo.jpg" alt="dexjoco logo" height="110">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.16257"><img src="https://img.shields.io/badge/arXiv-2605.16257-b31b1b?style=flat-square" alt="arXiv"></a>
  <a href="https://dexjoco.github.io/"><img src="https://img.shields.io/badge/GitHub-Page-FF6B00?style=flat-square&logo=github&logoColor=white" alt="Project Homepage"></a>
  <a href="https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-v1.1"><img src="https://img.shields.io/badge/🤗%20HF-Dataset-FFD21E?style=flat-square" alt="HF Dataset"></a>
  <a href="https://huggingface.co/papers/2605.16257"><img src="https://img.shields.io/badge/🤗%20HF-Paper-FFD21E?style=flat-square" alt="HF Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License"></a>
</p>

<p align="center">
  A MuJoCo benchmark for task-oriented dexterous manipulation.
</p>

# DexJoCo v1.1

**What’s new in v1.1**

- 🚀 **Easy installation** — Get started with `uv sync`.
- 🤖 **Easy policy evaluation** — Run policies with LeRobot’s `lerobot-eval`.
- ✅ **[100% replay success](#replay-success-rate)** — 2,200/2,200 episodes across all 11 tasks with corrected datasets.

**11 tasks across single-arm and bimanual manipulation**, including tool use, assembly, and device operation.

<table width="100%">
  <tr>
    <td align="center" width="20%">
      <img src="docs/pics/photograph_4.png" alt="Bimanual photograph" width="100%">
      <br><sub>Bimanual photograph</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/ipad_2.png" alt="Bimanual unlock iPad" width="100%">
      <br><sub>Bimanual unlock iPad</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/assembly_3.png" alt="Bimanual assembly" width="100%">
      <br><sub>Bimanual assembly</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/hanoi_3.png" alt="Bimanual hanoi" width="100%">
      <br><sub>Bimanual hanoi</sub>
    </td>
    <td align="center" width="20%">
      <img src="docs/pics/microwave_2.png" alt="Bimanual microwave" width="100%">
      <br><sub>Bimanual microwave</sub>
    </td>
  </tr>
</table>

<table width="100%">
  <tr>
    <td align="center" width="16.66%">
      <img src="docs/pics/water_plant_4.png" alt="Water plant" width="100%">
      <br><sub>Water plant</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/hammer_3.png" alt="Hammer nail" width="100%">
      <br><sub>Hammer nail</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/glass_1.png" alt="Fold glasses" width="100%">
      <br><sub>Fold glasses</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/tongs_3.png" alt="Pinch tongs" width="100%">
      <br><sub>Pinch tongs</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/bucket_4.png" alt="Pick bucket" width="100%">
      <br><sub>Pick bucket</sub>
    </td>
    <td align="center" width="16.66%">
      <img src="docs/pics/mouse_3.png" alt="Click mouse" width="100%">
      <br><sub>Click mouse</sub>
    </td>
  </tr>
</table>

## 📑 Table of contents

- [Installation](#installation)
- [Policy Evaluation](#evaluate-a-policy)
- [Datasets](#datasets)
- [Replay Validation](#replay-validation)
- [Replay Recording](#record-a-replay)
- [Replay Results](#replay-success-rate)
- [Data Format](#dataset-format)
- [TODO](#todo)
- [Acknowledgments](#acknowledgments)
- [License and Support](#license-and-support)
- [Citation](#citation)

<a id="installation"></a>

## 🚀 Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and **FFmpeg** using the [installation guide](docs/install_ffmpeg.md).

Get the LeRobot integration branch:

```bash
git clone --branch feat/lerobot https://github.com/brave-eai/dexjoco.git
cd dexjoco
uv sync
```

The examples below use `uv run`, which selects the project environment automatically.

<details>
<summary>Install with Conda</summary>

From the repository root:

```bash
conda env create -f environment.yml
conda activate dexjoco
```

With this environment active, run the examples with the `uv run` prefix removed.

</details>

<a id="evaluate-a-policy"></a>

## 🤖 Policy Evaluation

Select a task and a checkpoint trained with **matching action format, state features, camera names, and image resolution**.

| Model resource | Status |
| --- | --- |
| v1.1-compatible checkpoints | Coming soon |
| Model download and loading instructions | Coming soon |

### Local plugin

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=dexjoco --env.task=fold_glasses \
  --policy.path="$SINGLE_ARM_CHECKPOINT" --eval.batch_size=1

MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=dexjoco --env.task=bimanual_hanoi \
  --env.action_format=rotvec --policy.path="$BIMANUAL_CHECKPOINT" --eval.batch_size=1
```

### EnvHub

From the repository root, point `--env.hub_path` at the local checkout. DexJoCo's EnvHub publication is pending. The CLI accepts a JSON string for the environment options.

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=envhub --env.hub_path=. --env.trust_remote_code=true \
  --env.make_env_kwargs='{"task":"fold_glasses"}' \
  --policy.path="$SINGLE_ARM_CHECKPOINT" --eval.batch_size=1

MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=envhub --env.hub_path=. --env.trust_remote_code=true \
  --env.make_env_kwargs='{"task":"bimanual_hanoi","action_format":"rotvec"}' \
  --policy.path="$BIMANUAL_CHECKPOINT" --eval.batch_size=1
```

For `rand_full`, pass `--env.randomize=true` with the local plugin or add `"randomize":true` to the EnvHub JSON.

### Python

**Create a local environment directly.**

```python
from dexjoco.configs import DexJoCoEnvConfig

dex_cfg = DexJoCoEnvConfig(task="fold_glasses")
envs = dex_cfg.create_envs(n_envs=1)
```

**Load a configured environment through EnvHub.** Run this from the repository root so `hub_path="."` points to the local checkout.

```python
from dexjoco.configs import DexJoCoEnvConfig
from lerobot.envs.configs import HubEnvConfig

dex_cfg = DexJoCoEnvConfig(task="bimanual_hanoi", action_format="rotvec")
hub_cfg = HubEnvConfig(
    hub_path=".",
    trust_remote_code=True,
    make_env_kwargs=dex_cfg.to_make_env_kwargs(),
)
envs = hub_cfg.create_envs(n_envs=1)
```

**Load the default environment through the EnvHub shortcut.** From the repository root, this creates `water_plant` with `rand_obj` settings.

```python
from lerobot.envs import make_env

envs = make_env(".", n_envs=1, trust_remote_code=True)
```

Each example returns `envs["dexjoco"]`, a mapping from task IDs to vector environments. Call `close()` on each vector environment when finished. The [development guide](docs/development.md#test-envhub-locally) shows how to reset and step an EnvHub environment locally.

**Training and evaluation images can differ at the pixel level even with matching rendering settings.** Training decodes frames from the dataset's lossy AV1-encoded MP4 videos, while online evaluation uses RGB frames directly from MuJoCo. See [Reproducibility](docs/reproducibility.md#anti-aliasing-and-rendering).

<a id="datasets"></a>

## 📦 Datasets

| Dataset | Contents | Use | Download |
| --- | --- | --- | --- |
| v1.1 | Re-rendered demonstrations at 640 × 640 with MSAA disabled and saved initial scenes | Policy training, replay validation, and re-recording | [Hugging Face](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-v1.1) |

The v1.1 dataset contains the following groups:

| Group | Episodes per task | Replay recording settings |
| --- | ---: | --- |
| `rand_obj` | 100 | Recorded object placement; fixed cameras |
| `rand_full` | 100 | Recorded object placement; randomized cameras, lighting, and table textures |

Each dataset is organized as `<dataset_root>/{rand_obj,rand_full}/<task>`. **Keep `replay_initial_values.jsonl` alongside `meta/`, `data/`, and `videos/`** when copying a task dataset; replay uses it to restore the saved initial scene.

### Rotation-vector training

Prepare a training copy with `action_rotvec` stored as `action` so `lerobot-train` uses it as the target.

```bash
uv run dexjoco-prepare-rotvec \
  --dataset-dir datasets/lerobot_v1.1/rand_full/water_plant \
  --output-dir datasets/training_rotvec/rand_full/water_plant
```

Use the new directory as `--dataset.root` when training. Evaluate the resulting checkpoint with `--env.action_format=rotvec`. Use the original dataset for demonstration replay.

<a id="replay-validation"></a>

## ✅ Replay validation

To validate another local task dataset, select its task name and directory. The directory must include **`replay_initial_values.jsonl`**:

```bash
DATASET=datasets/lerobot_v1.1/rand_obj/water_plant

uv run dexjoco-replay --task water_plant --dataset-dir "$DATASET"
```

This replays all episodes without rendering, using the recorded quaternion actions, and prints the number of successful episodes.

**Replay commands automatically configure the CPU environment for reproducibility.** Both `dexjoco-replay` and `dexjoco-record-replay` set `OPENBLAS_CORETYPE=HASWELL` and `NPY_DISABLE_CPU_FEATURES` to disable selected AVX-512 features before importing NumPy. **No manual exports are needed when using these commands.** For integration into a Python process that imports NumPy first, see [CPU configuration](docs/reproducibility.md#cpu-configuration).

**Quaternion actions are the default for replay validation.** Converting orientations to rotation vectors and back can introduce small numerical differences that affect contact-sensitive trajectories. Consequently, replay with `--action-format rotvec` can fail on an episode that succeeds with quaternion actions. The default **`--extend-steps 60`** allows up to 60 extra control steps holding the final action.

<a id="record-a-replay"></a>

## 🎬 Record a replay

Save a new LeRobot dataset with rendered camera frames at a custom resolution using **`--render-hw HEIGHT WIDTH`**. Omit this option to use the environment default. **Choose a fresh output directory.** The following example records at 640 × 640 on a headless Linux machine with an EGL driver:

```bash
DATASET=datasets/lerobot_v1.1/rand_obj/water_plant

uv run dexjoco-record-replay \
  --task water_plant --dataset-dir "$DATASET" \
  --output-dir datasets/replays/rand_obj/water_plant \
  --render-hw 640 640
```

The recording command **defaults to EGL** for offscreen rendering. On a desktop with a working OpenGL display, optionally prefix the command with `MUJOCO_GL=glfw` to use GLFW instead. Camera images are rendered offscreen and saved to the output dataset.

For `rand_obj`, the recorded object placement is restored and the camera setup is fixed. For `rand_full`, point the command at the corresponding source and output directories and add `--randomize` (the default seed is 0) to vary cameras, lighting, and table textures. Episode *i* uses the base seed plus *i*. Both modes restore the saved initial scene.

**Recording stops on a failed episode by default.**

<a id="replay-success-rate"></a>

## 📊 Replay success rate

**All 2,200 episodes in the v1.1 release dataset replayed successfully across 11 tasks.** Validation used **quaternion actions**, **restored initial states**, and **up to 60 extra control steps holding the final action**.

| Dataset group | Episodes per task | Successful replays |
| --- | ---: | ---: |
| `rand_obj` | 100 | 1,100 / 1,100 (100%) |
| `rand_full` | 100 | 1,100 / 1,100 (100%) |
| **Total** | **200** | **2,200 / 2,200 (100%)** |

Rendering settings, CPU configuration, and image-storage considerations are documented in [Reproducibility](docs/reproducibility.md).

<a id="dataset-format"></a>

## 🗂️ Data Format

| LeRobot field | Single-arm dimensions | Bimanual dimensions | Contents |
| --- | ---: | ---: | --- |
| `observation.state` | 23 | 46 | Measured end-effector pose and 16 finger joints per arm |
| `action` | 23 | 46 | Target position, quaternion in `wxyz` order, and finger targets |
| `action_rotvec` | 22 | 44 | The same targets with orientation expressed as a rotation vector |
| `task` | | | Natural-language instruction, restored from task metadata when reading with `LeRobotDataset` |
| `observation.images.<camera>` | | | RGB frames. The main camera is `ego`; wrist and side views depend on the dataset. |

Both action fields and the robot state are stored as **float64**. State orientations also use **`wxyz` quaternions**. Bimanual state and actions share one order: **right arm pose, right hand joints, left arm pose, left hand joints**. Replay defaults to **`action`**. Pass `--action-format rotvec` to replay or re-record from `action_rotvec`.

**`replay_initial_values.jsonl`** sits alongside `meta/`, `data/`, and `videos/` in each task dataset. Each line contains an `episode_index` and a `replay_initial_values` object with task-specific values such as object poses and table height. Replay uses these values to restore the initial scene before applying the recorded actions.

<a id="todo"></a>

## TODO

- [ ] Run policies in simulation with `lerobot-rollout`.
- [ ] Record simulation episodes with `lerobot-record`.
- [ ] Teleoperate simulated robots with `lerobot-teleoperate`.
- [ ] Replay datasets in simulation with `lerobot-replay`.
- [ ] Serve DexJoCo environments with `lerobot-env-server`.

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

We thank the community members who identified and investigated issues:

- **[@BabyChemZ](https://github.com/BabyChemZ)** for finding that 157 of 1,101 raw demonstrations reached success only after holding the final recorded action for 1–39 additional steps ([issue #28](https://github.com/brave-eai/dexjoco/issues/28)).
- **[@chrockey](https://github.com/chrockey)** for identifying 42 `bimanual_unlock_ipad` demonstrations from the March 21, 2026 collection session that failed during replay, helping uncover a data-processing error in that batch ([issue #27](https://github.com/brave-eai/dexjoco/issues/27)).
- **[@Alioth7](https://github.com/Alioth7)** for investigating evaluation reproducibility and small pixel differences from EGL rendering with MSAA ([issue #24](https://github.com/brave-eai/dexjoco/issues/24)).

<a id="license-and-support"></a>

## 📄 License and Support

DexJoCo-owned code is released under the **[MIT License](LICENSE)**. Bundled third-party robot models retain their own licenses:

| Asset | License |
| --- | --- |
| [Franka Emika Panda](src/dexjoco/assets/robots/franka_emika_panda/README.md) | Apache-2.0 |
| [Wonik Allegro](src/dexjoco/assets/robots/wonik_allegro/README.md) | BSD-2-Clause |

For bugs and feature requests, [open an issue](https://github.com/brave-eai/dexjoco/issues).

For development setup and testing, see the [development guide](docs/development.md).

<a id="citation"></a>

## 📚 Citation

```bibtex
@misc{wang2026dexjocobenchmarktoolkittaskoriented,
      title={DexJoCo: A Benchmark and Toolkit for Task-Oriented Dexterous Manipulation on MuJoCo},
      author={Hanwen Wang and Weizhi Zhao and Xiangyu Wang and Siyuan Huang and He Lin and Boyuan Zheng and Rongtao Xu and Gang Wang and Yao Mu and He Wang and Lue Fan and Hongsheng Li and Zhaoxiang Zhang and Tieniu Tan},
      year={2026},
      eprint={2605.16257},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.16257},
}
```
