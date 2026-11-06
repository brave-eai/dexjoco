# DexJoCo v1.1

<p align="center"><img src="docs/pics/dexjoco_logo.jpg" alt="DexJoCo logo" height="110"></p>

[Project website](https://dexjoco.github.io/) · [Paper](https://arxiv.org/abs/2605.16257) · [Dataset collection](https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot) · [MIT license](LICENSE)

DexJoCo is a MuJoCo benchmark for task-oriented dexterous manipulation. This package brings its six single-arm and five bimanual tasks to Gymnasium and LeRobot. Use it to replay demonstrations from saved initial states, record new camera views, or evaluate a policy with `lerobot-eval`.

**Single-arm tasks**

<table>
  <tr>
    <td width="33%" align="center"><img src="docs/pics/water_plant_4.png" alt="water plant"><br><code>water_plant</code></td>
    <td width="33%" align="center"><img src="docs/pics/glass_1.png" alt="fold glasses"><br><code>fold_glasses</code></td>
    <td width="33%" align="center"><img src="docs/pics/mouse_3.png" alt="click mouse"><br><code>click_mouse</code></td>
  </tr>
  <tr>
    <td width="33%" align="center"><img src="docs/pics/tongs_3.png" alt="pinch tongs"><br><code>pinch_tongs</code></td>
    <td width="33%" align="center"><img src="docs/pics/bucket_4.png" alt="pick bucket"><br><code>pick_bucket</code></td>
    <td width="33%" align="center"><img src="docs/pics/hammer_3.png" alt="hammer nail"><br><code>hammer_nail</code></td>
  </tr>
</table>

**Bimanual tasks**

<table>
  <tr>
    <td width="33%" align="center"><img src="docs/pics/microwave_2.png" alt="bimanual microwave cook"><br><code>bimanual_microwave_cook</code></td>
    <td width="33%" align="center"><img src="docs/pics/ipad_2.png" alt="bimanual unlock ipad"><br><code>bimanual_unlock_ipad</code></td>
    <td width="33%" align="center"><img src="docs/pics/hanoi_3.png" alt="bimanual hanoi"><br><code>bimanual_hanoi</code></td>
  </tr>
  <tr>
    <td width="33%" align="center"><img src="docs/pics/assembly_3.png" alt="bimanual assembly"><br><code>bimanual_assembly</code></td>
    <td width="33%" align="center"><img src="docs/pics/photograph_4.png" alt="bimanual photograph"><br><code>bimanual_photograph</code></td>
  </tr>
</table>

## Installation

Run either set of commands from the repository root. Both install the checkout as an editable Python 3.12 package, including the DexJoCo environment plugin for LeRobot 0.6.1.

**With uv**, install the versions in the project lockfile and activate the environment. Video recording also requires FFmpeg on your system path.

```bash
uv sync --locked --python 3.12
source .venv/bin/activate
```

**With Conda**, create the environment from the supplied file. Conda provides Python and FFmpeg, and pip installs the project and its Python dependencies.

```bash
conda env create -f environment.yml
conda activate dexjoco-lerobot
```

Check that the replay command is available:

```bash
python -m dexjoco.replay_dataset --help
```

## Replay a demonstration dataset

Use a DexJoCo v1.1 LeRobot dataset for one task. Its directory must contain the LeRobot metadata and data together with `replay_initial_values.jsonl`, which holds the initial scene for each episode. Set the path to your local copy:

```bash
DATASET=datasets/lerobot_v1.1/rand_obj/water_plant

python -m dexjoco.replay_dataset \
  --task water_plant --dataset-dir "$DATASET" --n-envs 4
```

This replays all episodes without rendering, using the recorded quaternion actions. The default `--extend-steps 60` allows up to 60 extra control steps holding the final action. Some demonstrations reach the success condition shortly after the recorded actions end, as reported in [issue #28](https://github.com/brave-eai/dexjoco/issues/28).

### Replay success rate

The corrected v1.1 datasets passed replay validation across all 11 tasks with the default quaternion actions, restored initial states, and the 60-step extension.

| Dataset group | Episodes per task | Successful replays |
| --- | ---: | ---: |
| `rand_obj` | 100 | 1,100 / 1,100 (100%) |
| `rand_full` | 100 | 1,100 / 1,100 (100%) |
| **Total** | **200** | **2,200 / 2,200 (100%)** |

These results apply to the corrected v1.1 data and the validated runtime configuration. Earlier iPad demonstrations had a data-processing error reported in [issue #27](https://github.com/brave-eai/dexjoco/issues/27). Replay restores saved scene values, including task-specific initial hand joints where required. The success figures above were measured using `action` in quaternion format.

Replay entry points configure the following CPU settings before importing NumPy. Set them before starting Python when integrating replay into a process that imports NumPy first.

```bash
export OPENBLAS_CORETYPE=HASWELL
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"
```

## Record a replay at 640 × 640

To save a new LeRobot dataset with rendered camera frames, choose a fresh output directory. On a headless Linux machine with an EGL driver, run:

```bash
MUJOCO_GL=egl dexjoco-record-replay \
  --task water_plant --dataset-dir "$DATASET" \
  --output-dir datasets/replays/rand_obj/water_plant \
  --render-hw 640 640 --n-envs 2
```

For `rand_obj`, the recorded object placement is restored and the camera setup is fixed. For `rand_full`, point the command at the corresponding source and output directories and add `--randomize --seed 0` to vary cameras, lighting, and table textures. Episode *i* uses the base seed plus *i*, so changing the batch size preserves each episode’s visual seed. Both modes restore the saved initial scene.

Recording stops on a failed episode by default.

### Anti-aliasing and reproducibility

All task XMLs default to `offsamples=0`, disabling offscreen multisample anti-aliasing (MSAA). Replay and recording accept `--offsamples`; omitting it preserves the XML setting. Add `--offsamples 4` to record with four samples per pixel.

EGL rendering with MSAA enabled can produce small pixel differences between fresh rendering contexts. A visual policy can amplify these differences into different actions and success rates. Use matching anti-aliasing settings for dataset recording and policy evaluation. The v1.1 640 × 640 replay datasets were recorded with MSAA disabled. See [issue #24](https://github.com/brave-eai/dexjoco/issues/24).

That issue also reports a separate source of variation in OpenPI π0.5 inference through JAX/XLA. Its reported deterministic server configuration is:

```bash
export XLA_FLAGS="--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true --xla_gpu_autotune_level=0"
```

Apply these flags when starting the corresponding JAX policy server. Saved dataset videos use lossy compression, so decoded video pixels can differ from the original rendered frames.

## Observations and actions

| LeRobot field | Single arm | Bimanual | Contents |
| --- | ---: | ---: | --- |
| `observation.state` | 23 | 46 | Measured end-effector pose and 16 finger joints per arm |
| `action` | 23 | 46 | Target position, quaternion in `wxyz` order, and finger targets |
| `action_rotvec` | 22 | 44 | The same targets with orientation expressed as a three-component rotation vector |
| `observation.images.<camera>` | | | RGB frames. The main camera is `ego`; wrist and side views depend on the dataset. |

Both action fields and the robot state are stored as float64. Bimanual state and actions share one order: **right arm pose, right hand joints, left arm pose, left hand joints**. Replay defaults to `action`. Pass `--action-format rotvec` to replay or re-record from `action_rotvec`.

## Evaluate a policy

LeRobot discovers DexJoCo through the installed environment plugin. Select a task and a checkpoint trained with matching action and observation features. For a bimanual policy:

```bash
MUJOCO_GL=egl lerobot-eval \
  --env.type=dexjoco --env.task=bimanual_hanoi --env.bimanual=true \
  --policy.path=/path/to/pretrained_model \
  --eval.batch_size=1 --eval.n_episodes=5
```

For a single-arm policy, select its task and omit `--env.bimanual=true`. For a rotation-vector policy, also set `--env.action_format=rotvec`.

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
