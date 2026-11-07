# Development guide

Use Python 3.12 or newer. Install uv and FFmpeg as described in the [installation instructions](../README.md#installation), then run `uv sync` from the repository root. Run the development commands below from the same directory.

```bash
uv sync
```

## Where to make changes

| Area | Location |
| --- | --- |
| Task logic, reset, observations, and rendering | `src/dexjoco/envs/` |
| Task registration and LeRobot feature mapping | `src/dexjoco/envhub.py` and `src/dexjoco/configs.py` |
| Shared control and action conversion | `src/dexjoco/controllers.py`, `src/dexjoco/common.py`, and `src/dexjoco/actions.py` |
| MuJoCo scenes, meshes, and textures | `src/dexjoco/assets/` |
| Raw Zarr conversion and task presets | `src/dexjoco_data_converter/` |
| Dataset replay and recording | `src/dexjoco/replay_dataset.py` and `src/dexjoco/record_replay.py` |

For an environment change, start with the affected task and its tests. Use [water_plant](../src/dexjoco/envs/water_plant.py) as the reference for environment structure, asset layout, and tests. Keep code comments and docstrings in English.

1. Follow the [asset path rules](../src/dexjoco/assets/README.md) when adding or moving scenes. Preserve upstream asset licenses.
2. Restore saved scenes through `reset(options={"initial_state": ...})`, return the scene in `info["initial_state"]`, and report task success through `info["is_success"]`. Keep renderer creation lazy and release rendering resources in `close()`. Use `render_mode="none"` for physics-only runs.
3. Register new tasks in `envhub.py` and check that the configuration matches the actual action space and observations. Keep the main camera named `ego`, retain the established state and action layouts, and support the configured quaternion and rotation-vector action formats.
4. For tasks supported by the Zarr converter, add the conversion preset with the initial scene fields and keep its prompt consistent with the environment's `task_description`. Verify conversion and replay against the [dataset format](../README.md#dataset-format).

Keep machine-specific paths, datasets, checkpoints, and generated videos out of commits. Use temporary directories for test outputs and clean up files created during development.

## Development checks

```bash
uv run ruff check
uv run mypy
uv run pyright
uv run pytest
```

Ruff checks package source and project configuration. Mypy checks `src/`. Pyright checks the project, including tests, with the exclusions in `pyproject.toml`. Pytest runs the tests in `tests/`. Pass a test file to pytest to run just that file while developing. Tool settings are in [pyproject.toml](../pyproject.toml).

| Change | Tests to start with |
| --- | --- |
| Task behavior or initial scene restoration | The corresponding `tests/test_<task>.py` where available, plus `test_seed.py` or `test_initial_hand_joint.py` as applicable |
| Action representation or LeRobot configuration | `tests/test_actions.py`, `tests/test_env_config.py` |
| Plugin discovery, feature mapping, rollout, or rendering | `tests/test_lerobot_interface.py` |
| Zarr conversion or replay CLI behavior | `tests/test_zarr_loader.py`, `tests/test_replay_cli.py` |

Rendering tests use EGL by default when `MUJOCO_GL` is unset. See [Reproducibility](reproducibility.md) for replay and rendering settings.

## Validate with recorded demonstrations

Dataset replay tests are skipped unless you pass `--replay-dataset-root`. Point it at a local root containing both `rand_obj/<task>` and `rand_full/<task>`. The test checks that both action formats load at their stored precision and that two quaternion replay passes succeed and produce identical states and RGB frames.

```bash
MUJOCO_GL=egl uv run pytest tests/test_replay.py --replay-dataset-root datasets/lerobot_v1.1 --replay-n-envs=4
```

`--replay-n-envs` controls parallel environments and defaults to 4. Reduce it if rendering exhausts available memory. For a success-only check without rendering, use [dexjoco-replay](../README.md#replay-validation).

## Before committing

Use pytest to verify the final code you plan to commit. Clean up temporary files created during development.
