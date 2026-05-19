# Dexjoco Vive Bridge

This directory contains Dexjoco's own Vive/OpenVR bridge for streaming tracker
poses to the simulator over UDP.

It replaces the older vendored `Vive_Tracker` copy with a much smaller tool
that only implements the behavior Dexjoco needs:

- discover available OpenVR devices
- select a tracker by index or serial substring
- stream a `3x4 float64` pose matrix to UDP port `5012`

## Install

The default `dexjoco` environment does not install OpenVR. Add it only on the
machine that runs SteamVR:

```bash
pip install openvr
```

## List Devices

```bash
python teleoperation/vive_bridge/send_vive_pose.py --list-devices
```

If you know the exact device index, you can also use:

```bash
python teleoperation/vive_bridge/send_vive_pose.py --device-index 0
```

By default the sender streams to `127.0.0.1:5012` at `90 Hz`, which matches
Dexjoco's teleoperation wrappers.

## Stream Two Trackers For Bimanual Teleoperation

The bimanual teleop wrapper can also accept two tracker poses packed into one
UDP packet. To stream the first two discovered tracker devices:

```bash
python teleoperation/vive_bridge/send_vive_pose.py --two-trackers
```

You can also select them explicitly:

```bash
python teleoperation/vive_bridge/send_vive_pose.py \
  --device-index 3 \
  --second-device-index 4
```
