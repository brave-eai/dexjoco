# Repository Guidelines

use uv

## Project Structure & Module Organization

This is a multi-package robotics benchmark. Core MuJoCo environments, task configs, controllers, and simulation assets live in `dexjoco/dexjoco/`; task configs are in `tasks/`, and XML/mesh assets in `sim/envs/xmls/`. Top-level `configs/` contains policy evaluation YAMLs, while `scripts/` provides recording, replay, and smoke tests. `openpi/` contains policy training and serving; `dexjoco-data-converter/` handles Zarr and LeRobot conversion. `teleoperation/` contains device bridges and motion-capture tools; `docs/` holds integration references. Datasets and checkpoints are runtime data, not source files.

## Build, Test, and Development Commands

Create the simulation environment from the repository root:

```bash
conda env create -f environment-dexjoco.yaml
conda activate dexjoco
```

Install the core package with `pip install -e ./dexjoco` if needed. Run `python scripts/test_envs.py` for interactive smoke tests or `MUJOCO_GL=egl python scripts/test_headless_envs.py` for offscreen rendering (outputs go to `test_headless_videos/`). Collect demonstrations with `python scripts/record_demos_zarr.py --exp_name water_plant`; replay them with `python scripts/replay_demos_zarr.py --exp_name=water_plant --input_dir=./demos --out_dir=./replay_output`. OpenPI has separate setup (`cd openpi && bash install.bash`); see its README for training and serving.

## Coding Style & Naming Conventions

Follow the surrounding Python style: four-space indentation, `snake_case` modules/functions/variables, and `PascalCase` classes. Keep task implementations in `dexjoco/dexjoco/tasks/<task>/` and `sim/envs/`. Name config files after registered task IDs (for example, `water_plant.yaml`). Preserve asset-relative paths and XML references. No repository-wide formatter or linter is configured; match nearby code.

## Testing Guidelines

Core simulator checks are `scripts/test_envs.py` and `scripts/test_headless_envs.py`. Run the headless test with EGL after rendering, camera, or asset changes. OpenPI tests use pytest-style `*_test.py` names under `openpi/src/`. Add focused tests for behavior changes and report hardware or display prerequisites.

## Commit & Pull Request Guidelines

Recent commits use concise imperative subjects, sometimes with a PR number (for example, `Add ... (#17)`). Follow that style. Pull requests should summarize behavior changes, list validation commands and results, link related issues, and include screenshots or videos for visual changes. Do not commit datasets, checkpoints, or generated evaluation output.

## Configuration & Data Safety

Keep machine-specific paths, credentials, and generated artifacts out of tracked source. Evaluation and teleoperation may require display, device, or policy-server access; document those prerequisites.
