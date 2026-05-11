# dexjoco

This package contains DexJoCo's MuJoCo simulation environments, task
configuration layer, and demo storage utilities.

## Installation

From the repository root:

```bash
pip install -e ./dexjoco
```

## Explore the Environments

Use the top-level demo collection tool for interactive teleoperation data
collection:

```bash
python scripts/record_demos_zarr.py --exp_name water_plant
```

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

## Credits

- This simulation stack was originally built on top of work by [Kevin Zakka](https://kzakka.com/).
- DexJoCo environments adapt and extend that Gymnasium-based foundation.

## License

DexJoCo-owned code in this package is released under the MIT License. Bundled
third-party robot and hand assets under `dexjoco/sim/envs/xmls` retain their
own license terms.

## Notes

For machines that require EGL for offscreen rendering:

```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
