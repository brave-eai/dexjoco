## Teleoperation Interfaces

Please follow the instructions in [`Teleoperation_System_Tutorial.pdf`](Teleoperation_System_Tutorial.pdf) to assemble the teleoperation hardware and complete the required software setup. The [`GloveMount.STL`](GloveMount.STL) file provides the 3D-printable CAD model for mounting the glove and tracker.

This directory contains teleoperation interfaces and helper components that publish UDP
messages for DexJoCo's simulated data collection pipeline.

- [`vive_bridge/`](vive_bridge/): DexJoCo maintained OpenVR sender for Vive tracker poses.  Before pressing the keyboard key `;` to enable human intervention, keep your hand flat and align its pose with the simulated hand.
- [`rokoko/`](rokoko/): DexJoCo maintained Rokoko Studio bridge for forwarding
  canonicalized hand keypoints from another PC to the GeoRT/DexJoCo stack.
- [`GeoRT/`](GeoRT/): Third-party hand-retargeting component. This directory includes DexJoCo-specific Rokoko/UDP adaptations.



DexJoCo's simulation collector uses a few UDP payloads by default. The
providers in this directory are optional helpers around these default ports,
which can be changed in the corresponding teleoperation configs or scripts:

- `5012`: Vive tracker pose for single-arm and bimanual tasks.
- `5014`: right-hand or single-hand joint targets.
- `5016`: left-hand joint targets for bimanual tasks.
