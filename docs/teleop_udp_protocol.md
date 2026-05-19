# UDP Protocol for DexJoCo Teleoperation

This document provides the default UDP protocol settings for DexJoCo teleoperation. These settings can be modified according to the user's preferences and system requirements.

Dexjoco's teleoperation wrappers accept a small UDP protocol so the simulator
can stay decoupled from tracker, glove, or mocap implementations.

## Ports

- `5012`: tracker pose for single-arm and bimanual tasks
- `5014`: right-hand or single-hand joint targets
- `5016`: left-hand joint targets for bimanual tasks

## Tracker Payload

For single-arm teleoperation, port `5012` expects 12 `float64` values laid out
as a row-major `3x4` homogeneous transform:

```text
[r00 r01 r02 tx
 r10 r11 r12 ty
 r20 r21 r22 tz]
```

The simulator reconstructs this into a `4x4` matrix by appending
`[0, 0, 0, 1]`. Dexjoco records the current tracker pose continuously and
starts relative teleoperation when you press `;` in the MuJoCo window.

For bimanual teleoperation, the same port can also receive 24 `float64`
values: the first `3x4` block is the right tracker pose and the second `3x4`
block is the left tracker pose.

## Hand Payload

Ports `5014` and `5016` expect `float64` joint targets.

- Dexjoco reads the first 16 values from each packet.
- Shorter packets are zero-padded.
- Extra values are ignored.

The single-arm wrapper reads only `5014`. The bimanual wrapper reads `5014`
for the right hand and `5016` for the left hand.

## Local Controls

- `;`: toggle teleoperation on or off
- `R`: drop the current trajectory and request a reset

## Reference Providers

- [`teleoperation/vive_bridge`](../teleoperation/vive_bridge): Dexjoco's
  OpenVR tracker sender for port `5012`
- [`teleoperation/GeoRT`](../teleoperation/GeoRT): third-party, non-commercial
  hand-retargeting stack that can publish retargeted hand joints to `5014`
  and `5016`
