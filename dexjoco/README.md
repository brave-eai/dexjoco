# dexjoco

This package contains the core DexJoCo runtime, including MuJoCo simulation
environments, task configuration classes, demonstration recording and replay
utilities, and the OpenPI evaluation client used for policy rollouts.

Package layout:

| Path                     | Description                                                                       |
| ------------------------ | --------------------------------------------------------------------------------- |
| `dexjoco/tasks/`         | Task configuration classes and task-specific environment setup.                   |
| `dexjoco/sim/`           | MuJoCo simulation wrappers, controllers, and XML assets.                          |
| `dexjoco/data/`          | Zarr episode storage, video writing, depth capture, and replay support utilities. |
| `dexjoco_openpi_client/` | OpenPI evaluation client for DexJoCo policy rollout.                              |

## Installation

From the repository root:

```bash
conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

### Test the Installation

Run the interactive environment test from the repository root:

```bash
python scripts/test_envs.py
```

Run the headless rendering test with EGL:

```bash
python scripts/test_headless_envs.py
```

Both scripts iterate over the registered DexJoCo task configurations and step
each environment with zero actions. The headless test writes rendered videos to
`test_headless_videos/`.

Registered task names:

```text
bimanual_assembly
bimanual_hanoi
bimanual_microwave_cook
bimanual_photograph
bimanual_unlock_ipad
click_mouse
fold_glasses
hammer_nail
pick_bucket
pinch_tongs
water_plant
```

## Explore the Environments

Use the top-level demo collection tool for interactive teleoperation data
collection:

```bash
python scripts/record_demos_zarr.py --exp_name water_plant
```

The `--exp_name` value must match a registered task name. Interactive collection
uses the MuJoCo viewer and the task's teleoperation wrapper.

Recorded demonstrations can be replayed through the policy interface:

```bash
python scripts/replay_demos_zarr.py \
  --exp_name=water_plant \
  --input_dir=./demos \
  --out_dir=./replay_output \
  --randomize=True \
  --restore_state=True
```

Replay reads input folders containing `replay.zarr`, restores the recorded
initial scene state when available, executes the saved action sequence, and
writes replayed Zarr episodes and videos to the output directory. With
`--randomize=True`, replay can generate `rand_full` visual variants using preset
cameras, lighting, and texture randomization.

Common replay options:

| Option            | Default           | Description                                                                             |
| ----------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `--exp_name`      | `water_plant`     | Selects the task used to replay demonstrations.                                         |
| `--input_dir`     | `./`              | Directory containing recorded demo folders with `replay.zarr`.                          |
| `--out_dir`       | `./replay_output` | Output directory for replayed Zarr episodes and videos.                                 |
| `--randomize`     | `True`            | Enables replay-time `rand_full` visual randomization.                                   |
| `--restore_state` | `True`            | Restores initial table height and object poses from recorded `state[0]` when available. |
| `--save_failed`   | `False`           | Saves replay output even when the environment does not report success.                  |
| `--save_depth`    | `False`           | Saves depth arrays and depth videos alongside RGB videos.                               |

## Headless Mode

Set EGL for offscreen rendering:

```bash
export MUJOCO_GL=egl
```

Use `policy_mode=True` and `render_mode="rgb_array"` to construct headless
environments:

```python
TaskConfig.get_environment(policy_mode=True, render_mode="rgb_array", ...)
```

Common environment construction options:

| Option                    | Description                                                              |
| ------------------------- | ------------------------------------------------------------------------ |
| `policy_mode=True`        | Uses the policy action interface and disables interactive teleoperation. |
| `render_mode="rgb_array"` | Enables offscreen image observations for automated evaluation.           |
| `render_mode="human"`     | Opens the interactive MuJoCo viewer.                                     |
| `randomize=True`          | Enables the visual randomization regime used by `rand_full` evaluation.  |
| `randomize_dynamics=True` | Enables dynamics randomization when supported by the task.               |
| `seed=<int>`              | Sets the task environment seed.                                          |

## Policy Evaluation

The `dexjoco_openpi_client` package provides the DexJoCo-side client for OpenPI
policy evaluation. It adapts DexJoCo observations and actions to the OpenPI
WebSocket policy interface, including camera mapping, image resizing, prompt
injection, state slicing, and rotation-vector/quaternion action conversion.

### Rand Obj

This setting evaluates per-task policies with randomized object placement and
table height.

**serve policy**:

```bash
cd openpi
conda activate openpi
python scripts/serve_policy.py --port=8000 policy:checkpoint \
  --policy.config water_plant \
  --policy.dir ../checkpoints/pi05_ckpts/water_plant/<exp_name>/<step>
```

**evaluate policy**:

```bash
conda activate dexjoco
dexjoco-openpi-eval \
  --config=./configs/rand_obj/water_plant.yaml \
  --seed=0 \
  --port=8000
```

**config file**:

```yaml
env_name: water_plant
camera_mapping:
  base: front
  wrist: wrist
prompt: "Grasp the watering can and apply water to the plant."
robot_type: single_arm
```

### Rand Full

This setting evaluates per-task policies with object placement, table height,
camera pose, illumination, and tabletop texture randomization.

**serve policy**:

Use the same `serve_policy` command structure with the matching `rand_full`
policy config and checkpoint, such as `water_plant_rand_full`.

**evaluate policy**:

```bash
dexjoco-openpi-eval \
  --config=./configs/rand_full/bimanual_microwave_cook.yaml \
  --seed=0 \
  --port=8000 \
  --rand-full
```

**config file**:

```yaml
env_name: water_plant
camera_mapping:
  base: random_camera
  wrist: wrist
prompt: "Grasp the watering can and apply water to the plant."
robot_type: single_arm
```

### Multi-Task

This setting evaluates policies trained jointly across the DexJoCo task suite
with a unified observation and state representation.

**serve policy**:

Use the same `serve_policy` command structure with the multi-task policy config
and checkpoint, such as `multi_task`.

**evaluate policy**:

Pass `--pad-state-dim46` to pad single-arm state vectors to the 46-dimensional
dual-arm state layout used by the multi-task training data.

```bash
dexjoco-openpi-eval \
  --config=./configs/multi_task/water_plant.yaml \
  --seed=0 \
  --port=8000 \
  --pad-state-dim46
```

**config file**:

Multi-task configs use a unified observation layout across DexJoCo tasks. For
single-arm tasks, both `wrist_left` and `wrist_right` are mapped to the same
environment `wrist` camera.

```yaml
env_name: water_plant
camera_mapping:
  base: front
  wrist_left: wrist
  wrist_right: wrist
prompt: "Grasp the watering can and apply water to the plant."
robot_type: single_arm
```

### iPad Reasoning

This setting evaluates language-conditioned iPad unlock behavior with digit,
word, and arithmetic password instructions.

**serve policy**:

Use the same `serve_policy` command structure with an iPad unlock policy
checkpoint.

**evaluate policy**:

Use `--output` to keep results from different prompt and password configs in
separate directories.

```bash
dexjoco-openpi-eval \
  --config=./configs/ipad_reasoning/2+2.yaml \
  --seed=0 \
  --port=8000 \
  --output=./outputs/ipad_reasoning_2_plus_2_seed0
```

**config file**:

iPad reasoning configs target `bimanual_unlock_ipad` with prompt variants such
as digits, words, and arithmetic expressions. The optional `password` list sets
the target iPad password and is used only by the iPad unlock environment.

```yaml
env_name: bimanual_unlock_ipad
camera_mapping:
  base: ego
  wrist_left: wrist_left
  wrist_right: wrist_right
prompt:
  "Grasp the iPad and enter the result of 2+2 as the password to unlock the
  device."
password: [4]
robot_type: dual_arm
```

Evaluation videos and success-rate marker files are written under `outputs/` by
default. Use `--output` to select a different output directory.

Evaluation configs live under `configs/rand_obj/`, `configs/rand_full/`,
`configs/multi_task/`, and `configs/ipad_reasoning/`. Each config defines the
DexJoCo task, camera mapping, language prompt, and robot layout. Single-arm
configs set `robot_type` to `single_arm` and use a `base` camera with one wrist
camera. Dual-arm configs set `robot_type` to `dual_arm` and use `base`,
`wrist_left`, and `wrist_right` cameras.

`dexjoco-openpi-eval` supports the following options:

| Option                            | Default        | Description                                                                              |
| --------------------------------- | -------------- | ---------------------------------------------------------------------------------------- |
| `--config PATH`                   | Required       | Evaluation YAML under one of the supported `configs/` subdirectories.                    |
| `--seed INT`                      | `0`            | Random seed for NumPy and Python random state.                                           |
| `--rand-full`                     | `False`        | Enables the `rand_full` visual randomization regime.                                     |
| `--randomize-dynamics`            | `False`        | Enables dynamics randomization.                                                          |
| `--port INT`                      | `8000`         | OpenPI WebSocket policy server port.                                                     |
| `--host HOST`                     | `0.0.0.0`      | Host address used by the OpenPI WebSocket client.                                        |
| `--output PATH`                   | Auto-generated | Output directory for videos and success-rate marker files.                               |
| `--render-mode {rgb_array,human}` | `rgb_array`    | DexJoCo rendering mode.                                                                  |
| `--replan-ratio FLOAT`            | `0.8`          | Fraction of the OpenPI action horizon to execute before requesting a fresh action chunk. |
| `--episodes INT`                  | `50`           | Number of evaluation episodes to run.                                                    |
| `--pad-state-dim46`               | `False`        | Pads the policy state representation to 46 dimensions for compatibility.                 |
| `--record-pressed-digits BOOL`    | Task-dependent | Controls whether iPad digit inputs are recorded in episode output names.                 |

## Credits

- This simulation stack was originally built on top of work by
  [Kevin Zakka](https://kzakka.com/).
- DexJoCo environments adapt and extend that Gymnasium-based foundation.

## License

DexJoCo-owned code in this package is released under the MIT License. Bundled
third-party robot and hand assets under `dexjoco/sim/envs/xmls` retain their own
license terms.
