# dexjoco_sim

This package contains the MuJoCo simulation environments used by Dexjoco.

## Installation

From the repository root:

```bash
cd dexjoco_sim
pip install -e .
pip install -r requirements.txt
```

## Explore the Environments

Use the top-level demo collection tool to run the maintained tasks:

```bash
cd ..
python scripts/record_demos_zarr.py --exp_name water_plant
```

## Credits

- This simulation stack was originally built on top of work by [Kevin Zakka](https://kzakka.com/).
- The current Dexjoco environments adapt and extend that Gymnasium-based foundation.

## Notes

For CPU-only machines that need EGL:

```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```
