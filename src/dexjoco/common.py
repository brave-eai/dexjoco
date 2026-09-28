"""Shared package paths and simulation timing constants."""

from pathlib import Path

_ASSET_ROOT = Path(__file__).resolve().parent / "assets"
_CONTROL_TIMESTEP = 0.02
_PHYSICS_TIMESTEP = 0.002

_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
_ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
_FINGER_JOINTS = (
    "ffj0", "ffj1", "ffj2", "ffj3",
    "mfj0", "mfj1", "mfj2", "mfj3",
    "rfj0", "rfj1", "rfj2", "rfj3",
    "thj0", "thj1", "thj2", "thj3",
)
_FINGER_ACTUATORS = (
    "ffa0", "ffa1", "ffa2", "ffa3",
    "mfa0", "mfa1", "mfa2", "mfa3",
    "rfa0", "rfa1", "rfa2", "rfa3",
    "tha0", "tha1", "tha2", "tha3",
)
