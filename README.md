## DexJoCo

DexJoCo is a MuJoCo-based simulation benchmark for dexterous hands

## Installation

```bash
conda env create -f environment-dexjoco.yml
conda activate dexjoco
```


## Teleoperation Components

teleoperation helpers live under
[`teleoperation/`](teleoperation):

- [`teleoperation/vive_bridge`](teleoperation/vive_bridge): DexJoCo-maintained
  OpenVR sender for Vive tracker poses on UDP port `5012`
- [`teleoperation/rokoko`](teleoperation/rokoko): DexJoCo-maintained Rokoko
  Studio bridge for forwarding canonicalized raw hand keypoints from another PC
  to the GeoRT/DexJoCo stack
- [`teleoperation/GeoRT`](teleoperation/GeoRT): third-party GeoRT code kept
  in-repo for non-commercial research use, including DexJoCo's Rokoko-to-UDP
  hand retargeting scripts for ports `5014` and `5016`

The simulator itself only depends on the UDP packets described in
[`docs/teleop_udp_protocol.md`](docs/teleop_udp_protocol.md).

## Quick Start

```bash
python scripts/record_demos_zarr.py \
  --exp_name water_plant \
  --successes_needed 1 
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

DexJoCo-owned code in this repository is released under the
[`MIT License`](LICENSE).

Bundled third-party components keep their separate license terms. In
particular, [`teleoperation/GeoRT`](teleoperation/GeoRT) remains under its
upstream non-commercial license and is not covered by the MIT License.
