"""Shared package paths and simulation timing constants."""

from pathlib import Path

import numpy as np

ASSET_ROOT = Path(__file__).resolve().parent / "assets"
CONTROL_TIMESTEP = 0.02
PHYSICS_TIMESTEP = 0.002

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
ARM_HOME = np.array([0.0, -0.785, 0.0, -2.35, 0.0, 1.57, np.pi / 4])
FINGER_HOME = np.zeros(16, dtype=np.float32)
FINGER_HOME[12] = 0.263
# TCP workspace envelope: robot base [-0.8, 0, 0.924] (water_plant.xml) plus and
# minus the summed link offsets from base to site "attachment_site"
# (attachment 0.107 + link7 0.088 + link5 0.3928 + link4 0.0825 + link3 0.316
# + link1 0.333 + link0 0.05 = 1.3693). By the triangle inequality the TCP
# distance from the base never exceeds that sum in any arm pose. Recompute if
# the robot mount or the arm kinematic chain changes.
TCP_LOW = np.array([-2.16926233, -1.36926233, -0.44526233])
TCP_HIGH = np.array([0.56926233, 1.36926233, 2.29326233])
# MuJoCo enforces joint limits as soft constraints, so observed finger qpos
# can overshoot jnt_range; widen observation bounds by this margin.
FINGER_QPOS_MARGIN = 0.1
TABLE_TEXTURES = (
    "table_bamboo",
    "table_blue-wood",
    "table_brass-ambra",
    "table_ceramic",
    "table_cream-plaster",
    "table_dark_wood_planks_2",
    "table_dark-wood",
    "table_gray-plaster",
    "table_gray_wood_planks",
    "table_light-wood",
    "table_metal",
    "table_pink-plaster",
    "table_red-wood",
    "table_legs_metal",
    "table_steel-scratched",
    "table_walnut_wood_grain",
    "table_warm_wood_grain_2",
    "table_white-plaster",
    "table_wood_grain_1",
    "table_yellow-plaster",
)
FINGER_JOINTS = (
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
)
FINGER_ACTUATORS = (
    "ffa0", "ffa1", "ffa2", "ffa3",
    "mfa0", "mfa1", "mfa2", "mfa3",
    "rfa0", "rfa1", "rfa2", "rfa3",
    "tha0", "tha1", "tha2", "tha3",
)
