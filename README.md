## dexjoco

Dexjoco is a MuJoCo-based simulation benchmark for dexterous hands and robotic arms. The default public research setup is simulation-only and does not require RealSense, SpaceMouse, OpenVR, Franka robot servers, JAX, or TensorFlow.

## Installation

```bash
conda env create -f environment-dexjoco.yml
conda activate dexjoco
```

The environment pins MuJoCo 3.4.0 and Gymnasium 1.0.0.

## Optional Teleoperation Components

The default research environment is simulation-first and does not install
SteamVR or OpenVR tooling. Optional teleoperation helpers live under
[`teleoperation/`](teleoperation):

- [`teleoperation/vive_bridge`](teleoperation/vive_bridge): Dexjoco-maintained
  OpenVR sender for Vive tracker poses on UDP port `5012`
- [`teleoperation/rokoko`](teleoperation/rokoko): Dexjoco-maintained Rokoko
  Studio bridge for forwarding canonicalized raw hand keypoints from another PC
  to the GeoRT/Dexjoco stack
- [`teleoperation/GeoRT`](teleoperation/GeoRT): third-party GeoRT code kept
  in-repo for non-commercial research use, including Dexjoco's Rokoko-to-UDP
  hand retargeting scripts for ports `5014` and `5016`

The simulator itself only depends on the UDP packets described in
[`docs/teleop_udp_protocol.md`](docs/teleop_udp_protocol.md).

## Quick Start

Dexjoco keeps large runtime assets out of Git. Before running simulation tasks, restore the simulation XML and asset bundle:

```bash
python scripts/download_assets.py --bundle sim-xmls
```

If you plan to use GeoRT-based teleoperation, also restore the GeoRT runtime assets:

```bash
python scripts/download_assets.py --bundle geort-runtime-assets
```

Additional details live in [`docs/assets.md`](docs/assets.md).

```bash
python scripts/record_demos_zarr.py \
  --exp_name water_plant \
  --successes_needed 1 \
  --show_sim_cameras False
```

## Demo Format

`scripts/record_demos_zarr.py` writes each successful demo as:

```text
<out_dir>/<exp_name>_demo_<index>_<timestamp>/
  replay.zarr/
  videos/<camera_key>.mp4
```

The Zarr replay buffer and H.264 video writer live in the local
`dexjoco_data` package used by the demo collection tools.

## License

The root of this repository is released under the
[`Dexjoco Research License`](LICENSE). It is source-available for
non-commercial scientific research use only and is not an open-source
license.

Bundled third-party components may keep separate license terms. In
particular, [`teleoperation/GeoRT`](teleoperation/GeoRT) remains under its own
third-party non-commercial license.
