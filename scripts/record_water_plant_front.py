"""Record three simulated seconds: uv run scripts/record_water_plant_front.py."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.common import CONTROL_TIMESTEP
from dexjoco.envhub import make_env


def hold_action(proprio: np.ndarray) -> np.ndarray:
    """Convert batched proprioception into absolute rotvec action targets."""
    rotvec = Rotation.from_quat(proprio[:, 3:7], scalar_first=True).as_rotvec()
    return np.concatenate((proprio[:, :3], rotvec, proprio[:, 7:]), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("water_plant_front.mp4"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Match the control rate so video duration equals simulation duration.
    fps = round(1 / CONTROL_TIMESTEP)
    env = make_env(n_envs=1)["dexjoco"][0]  # Task 0 is water_plant.
    try:
        obs, _ = env.reset(seed=0)
        # Actions are absolute targets: hold the initial TCP pose and finger joints.
        action = hold_action(obs["proprio"])
        height, width = obs["images"]["front"][0].shape[:2]
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
        )
        try:
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video output: {args.output}")
            for _ in range(round(3 / CONTROL_TIMESTEP)):
                obs, _, terminated, truncated, _ = env.step(action)
                writer.write(cv2.cvtColor(obs["images"]["front"][0], cv2.COLOR_RGB2BGR))
                if terminated[0] or truncated[0]:
                    # EnvHub uses SAME_STEP autoreset; obs is already reset.
                    action = hold_action(obs["proprio"])
        finally:
            writer.release()
    finally:
        env.close()

    print(f"Saved 3-second front camera video to {args.output.resolve()}")


if __name__ == "__main__":
    main()
