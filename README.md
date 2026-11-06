# DexJoCo v1.1

DexJoCo is a MuJoCo benchmark for task-oriented dexterous manipulation.
DexJoCo v1.1 integrates with LeRobot for straightforward installation, policy evaluation, and demonstration replay.

## Installation

### uv

Install FFmpeg using the [installation guide](docs/install_ffmpeg.md), then run the following commands from the repository root:

```bash
uv sync
```

### Conda

```bash
conda env create -f environment.yml
conda activate dexjoco
```

The examples below use `uv run`, which selects the project environment automatically. If you installed with Conda, activate `dexjoco` and run the same commands with the `uv run` prefix removed.

## Dataset format

Each task is stored as a LeRobot dataset, with an additional `replay_initial_values.jsonl` file containing the saved initial scene for every episode. Keep this file alongside the dataset’s `meta/`, `data/`, and `videos/` directories when copying a dataset.

The v1.1 datasets are organized as `<dataset_root>/{rand_obj,rand_full}/<task>`. Robot observations and both action representations are described in [Observations and actions](#observations-and-actions).

## Replay validation

Use a DexJoCo v1.1 LeRobot dataset for one task. Its directory must contain the LeRobot metadata and data together with `replay_initial_values.jsonl`, which holds the initial scene for each episode. Set the path to your local copy:

```bash
DATASET=datasets/lerobot_v1.1/rand_obj/water_plant

uv run dexjoco-replay --task water_plant --dataset-dir "$DATASET"
```

This replays all episodes without rendering, using the recorded quaternion actions, and prints the number of successful episodes. The command exits with status 1 and lists failed episode indices if any episode fails.

Quaternion actions are the default for replay validation. Converting orientations to rotation vectors and back can introduce small numerical differences that affect contact-sensitive trajectories. Consequently, replay with `--action-format rotvec` can fail on an episode that succeeds with quaternion actions. The default `--extend-steps 60` allows up to 60 extra control steps holding the final action. Some demonstrations reach the success condition shortly after the recorded actions end, as reported in [issue #28](https://github.com/brave-eai/dexjoco/issues/28).

## Evaluate a policy

LeRobot discovers DexJoCo through the installed environment plugin. Select a task and a checkpoint trained with matching action and observation features. For a bimanual policy:

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --env.type=dexjoco --env.task=bimanual_hanoi \
  --policy.path=/path/to/pretrained_model
```

The task determines the arm configuration, action dimensions, and wrist cameras automatically. For a rotation-vector policy, also set `--env.action_format=rotvec`.

## Record a replay

Save a new LeRobot dataset with rendered camera frames at a custom resolution using `--render-hw HEIGHT WIDTH`. Omit this option to use the environment default. Choose a fresh output directory. The following example records at 640 × 640 on a headless Linux machine with an EGL driver:

```bash
uv run dexjoco-record-replay \
  --task water_plant --dataset-dir "$DATASET" \
  --output-dir datasets/replays/rand_obj/water_plant \
  --render-hw 640 640
```

The recording command defaults to EGL for offscreen rendering. On a desktop with a working OpenGL display, optionally prefix the command with `MUJOCO_GL=glfw` to use GLFW instead. Camera images are rendered offscreen and saved to the output dataset.

For `rand_obj`, the recorded object placement is restored and the camera setup is fixed. For `rand_full`, point the command at the corresponding source and output directories and add `--randomize` (the default seed is 0) to vary cameras, lighting, and table textures. Episode *i* uses the base seed plus *i*, so changing the batch size preserves each episode’s visual seed. Both modes restore the saved initial scene.

Recording stops on a failed episode by default.

## Replay success rate

The corrected v1.1 datasets passed replay validation across all 11 tasks with the default quaternion actions, restored initial states, and the 60-step extension.

| Dataset group | Episodes per task | Successful replays |
| --- | ---: | ---: |
| `rand_obj` | 100 | 1,100 / 1,100 (100%) |
| `rand_full` | 100 | 1,100 / 1,100 (100%) |
| **Total** | **200** | **2,200 / 2,200 (100%)** |

These results apply to the corrected v1.1 data and the validated runtime configuration. Replay restores saved scene values. The success figures above were measured using `action` in quaternion format.

Replay entry points configure the following CPU settings before importing NumPy. Set them before starting Python when integrating replay into a process that imports NumPy first.

```bash
export OPENBLAS_CORETYPE=HASWELL
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
```

## Anti-aliasing and reproducibility

All task XMLs default to `offsamples=0`, disabling offscreen multisample anti-aliasing (MSAA). Replay and recording accept `--offsamples`; omitting it preserves the XML setting. Add `--offsamples 4` to record with four samples per pixel.

EGL rendering with MSAA enabled can produce small pixel differences between fresh rendering contexts. A visual policy can amplify these differences into different actions and success rates. Use matching anti-aliasing settings for dataset recording and policy evaluation. The v1.1 640 × 640 replay datasets were recorded with MSAA disabled. See [issue #24](https://github.com/brave-eai/dexjoco/issues/24).

Dataset videos use lossy compression to keep storage requirements manageable. Training therefore receives decoded video frames, while online inference receives freshly rendered frames. Even with identical rendering settings, these images can differ at the pixel level. Lossless storage would substantially increase dataset size.

## Observations and actions

| LeRobot field | Single-arm dimensions | Bimanual dimensions | Contents |
| --- | ---: | ---: | --- |
| `observation.state` | 23 | 46 | Measured end-effector pose and 16 finger joints per arm |
| `action` | 23 | 46 | Target position, quaternion in `wxyz` order, and finger targets |
| `action_rotvec` | 22 | 44 | The same targets with orientation expressed as a three-component rotation vector |
| `observation.images.<camera>` | | | RGB frames. The main camera is `ego`; wrist and side views depend on the dataset. |

Both action fields and the robot state are stored as float64. State orientations also use `wxyz` quaternions. Bimanual state and actions share one order: **right arm pose, right hand joints, left arm pose, left hand joints**. Replay defaults to `action`. Pass `--action-format rotvec` to replay or re-record from `action_rotvec`.

## Citation

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
