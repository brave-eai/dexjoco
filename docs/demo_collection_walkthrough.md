# Simulation Demo Collection Walkthrough
Optional UDP teleoperation providers are documented in
[`teleop_udp_protocol.md`](teleop_udp_protocol.md).
The main workflow is:

1. choose a simulation task
2. run `scripts/record_demos_zarr.py`
3. teleoperate in MuJoCo and save successful episodes as Zarr demos

Task names:

- `bimanual_assembly`
- `bimanual_hanoi`
- `bimanual_microwave_cook`
- `bimanual_photograph`
- `bimanual_unlock_ipad`
- `click_mouse`
- `fold_glasses`
- `hammer_nail`
- `pick_bucket`
- `pinch_tongs`
- `water_plant`

## Environment

Use the `dexjoco` conda environment:

```bash
conda run -n dexjoco python scripts/record_demos_zarr.py --exp_name=water_plant
```

If you prefer to enter the environment first:

```bash
conda activate dexjoco
cd scripts
python record_demos_zarr.py --exp_name=water_plant
```

## Recommended command

From the repository root:

```bash
conda run -n dexjoco python scripts/record_demos_zarr.py \
  --exp_name=water_plant \
  --successes_needed=20 \
  --randomize=True
```

Useful flags:

- `--exp_name`: one of the task names above
- `--successes_needed`: stop automatically after this many successful demos
- `--randomize=True`: keep simulation domain randomization enabled during collection
- `--render_mode=human`: open the MuJoCo viewer for teleoperation
- `--show_sim_cameras=True`: show the OpenCV camera grid window
- `--out_dir=PATH`: choose where the collected demos are written

## During collection

- successful episodes are saved automatically
- failed episodes are discarded automatically
- press `R` in the viewer loop to drop the current trajectory and reset
- teleop wrappers keep the current end-effector pose when you are not actively intervening

Each successful demo is written to its own directory:

```text
<out_dir>/<exp_name>_demo_<index>_<timestamp>/
  replay.zarr/
  videos/
```

## Notes

- The task configs under `dexjoco.tasks.*.config` are centered on simulated teleoperation capture.
- Domain randomization is preserved through the `randomize` argument in each task environment.
