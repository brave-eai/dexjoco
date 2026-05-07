# Mocap Instructions

This Dexjoco release keeps only the Rokoko-facing GeoRT mocap entry points used by the teleoperation stack:

- `rokoko_retarget_send_left.py`
- `rokoko_retarget_send_right.py`
- `rokoko_evaluation.py`

MediaPipe and Manus-specific support have been removed from this repository snapshot.

If you want to sanity-check a trained checkpoint with live Rokoko input, run one of the `rokoko_retarget_send_*` scripts after your Rokoko bridge is already forwarding canonicalized keypoints to the Dexjoco host.
