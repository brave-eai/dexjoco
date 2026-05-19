## Teleoperation Providers

please follow the instructions in [`Teleoperation_System_Tutorial.pdf`](Teleoperation_System_Tutorial.pdf) to assemble the teleoperation hardware and complete the required software setup. The [`GloveMount.STL`](GloveMount.STL) file provides the 3D-printable CAD model for mounting the glove and tracker.

This directory contains teleoperation providers that can publish UDP
messages for DexJoCo's simulated data-collection pipeline.

- [`vive_bridge/`](vive_bridge/): DexJoCo-maintained OpenVR sender for Vive tracker poses.
- [`rokoko/`](rokoko/): DexJoCo-maintained Rokoko Studio bridge for forwarding
  canonicalized hand keypoints from another PC to the GeoRT/DexJoCo stack.
- [`GeoRT/`](GeoRT/): third-party hand-retargeting component. This directory includes DexJoCo-specific Rokoko/UDP adaptations.

DexJoCo's simulation collector only depends on the UDP payloads documented in
[`../docs/teleop_udp_protocol.md`](../docs/teleop_udp_protocol.md). The
providers in this directory are optional helpers around that protocol. 
