# Custom Policy Integration

This document describes how to adapt a custom policy to DexJoCo and how to
evaluate it against the benchmark environments.

DexJoCo exposes one environment contract: observations are collected from the
simulation, passed into a policy, converted into actions, and executed back in
the simulator. The current OpenPI client in
[`dexjoco/dexjoco_openpi_client`](../dexjoco/dexjoco_openpi_client) is the
primary reference for this integration path. The reference code under
[`docs/reference_code`](reference_code) shows two additional implementation
patterns: multi-frame observation history and LeRobot `async_inference` reuse.

## Policy Protocol

### Observation

A policy should consume the observation fields defined by the task config. The
core fields are:

- camera images mapped through `camera_mapping`
- `state`
- `prompt`

For OpenPI-style evaluation, the wrapper converts DexJoCo observations into the
policy input format expected by the model server. The wrapper also handles image
resize, image dtype conversion, and camera key remapping.

The `state` field contains both proprioception and privileged environment
information. Policy inputs should use proprioception only:

- single-arm: the first 23 dimensions
- dual-arm: the first 46 dimensions

Privileged state includes task-specific environment variables such as object
poses and table parameters. Those values are useful for replay and reset
restoration, but they should not be used as policy inputs.

### Action

DexJoCo policy actions use rotation vectors for end-effector orientation:

- single-arm policy action: 22 dimensions
  - `[xyz(3), rotvec(3), hand(16)]`
- dual-arm policy action: 44 dimensions
  - `[r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)]`

The simulator executes quaternion pose formats internally:

- single-arm environment action: 23 dimensions
  - `[xyz(3), quat(4), hand(16)]`
- dual-arm environment action: 46 dimensions
  - `[r_xyz(3), r_quat(4), l_xyz(3), l_quat(4), r_hand(16), l_hand(16)]`

The conversion boundary is therefore the policy wrapper. The policy should emit
rotvec actions; the environment wrapper should convert them to quaternions
before `env.step()`.

## Action Chunks

OpenPI evaluation is the main reference for chunked action execution. The policy
server returns a horizon of actions, and the DexJoCo client executes the buffer
while requesting the next plan before the buffer runs dry.

1. send the latest observation to inference
2. receive an action chunk
3. execute the buffered actions in timestamp order
4. request a new chunk before the buffer is exhausted

The OpenPI client under
[`dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py`](../dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py)
implements this loop directly.

The important parts are:

- action chunks are aligned to observation timestamps
- buffered actions are executed before requesting new inference
- overlapping chunks are blended in the overlap window
- replan is triggered when the buffered horizon drops below a threshold

This pattern avoids blocking the control loop on inference latency and gives the
policy a smoother transition between successive action plans.

## Multi-Frame Observation History

Some policies benefit from a short observation window rather than a single
frame. The reference implementation under
[`docs/reference_code/dexjoco_dp_eval/dp_dexjoco_env.py`](reference_code/dexjoco_dp_eval/dp_dexjoco_env.py)
uses a `deque` to maintain a fixed-length history:

- `reset()` seeds the queue with the initial observation
- `step()` appends the latest observation
- `get_obs()` stacks the frames along a new history dimension

This keeps history management local to the wrapper and leaves the policy code
focused on model inference. The same approach can be extended to longer windows
by increasing the deque length and the stack logic.

## Reusing `async_inference`

The LeRobot reference under
[`docs/reference_code/dexjoco_lerobot_client`](reference_code/dexjoco_lerobot_client)
shows how to reuse LeRobot's built-in `async_inference` support around DexJoCo.
See the official documentation at
[https://huggingface.co/docs/lerobot/async](https://huggingface.co/docs/lerobot/async).

The key idea is to separate the control loop from observation transport:

- the simulator wrapper produces observations and actions
- the `RobotClient` handles chunked action delivery and observation timing
- a background sender thread keeps observation transport off the critical path

That pattern is useful when a policy already fits the LeRobot-style async
inference API. DexJoCo also needs a rewritten robot client so the background
observation sender can run as `self._obs_sender_thread` without blocking the
control loop.

## Why the DP and LeRobot Reference Code Is Not Open-Sourced Yet

The Diffusion Policy and LeRobot paths are kept as reference code because they
depend on substantial adaptation work:

- the Diffusion Policy training stack requires large changes to support DexJoCo
  observation and action layouts
- the LeRobot path requires source changes and monkey patching before it can be
  used with the DexJoCo simulator

For that reason, the repository exposes the stable DexJoCo and OpenPI
integration path directly, while the Diffusion Policy and LeRobot material
remains as reference code for implementation patterns.

## Evaluation Entry Points

Use the OpenPI client for the current evaluation path:

- [`dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py`](../dexjoco/dexjoco_openpi_client/dexjoco_openpi_env.py)
- [`dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py`](../dexjoco/dexjoco_openpi_client/eval_dexjoco_openpi.py)
- [`configs/rand_obj/*.yaml`](../configs/rand_obj)
- [`configs/rand_full/*.yaml`](../configs/rand_full)

The reference implementations remain available for studying multi-frame history,
action chunking, and async inference reuse:

- [`docs/reference_code/dexjoco_dp_eval`](reference_code/dexjoco_dp_eval)
- [`docs/reference_code/dexjoco_lerobot_client`](reference_code/dexjoco_lerobot_client)
