"""
Provide state, action, timestamp and replay initial positions for lerobot dataset
"""

from pathlib import Path

import numpy as np
import zarr


def load_zarr_episode(episode_dir: Path, action_dim: int, state_dim: int):
    root = zarr.open_group(str(episode_dir / "replay.zarr"), mode="r")
    arrays = {
        key: np.asarray(root["data"][key][:])
        for key in ("action_rotvec", "state", "timestamp")
    }

    action = arrays["action_rotvec"]
    state = arrays["state"][:, 0, :state_dim]
    replay_initial_positions = arrays["state"][0, 0, state_dim:]

    assert action.shape[0] == state.shape[0]

    assert action.ndim == 2 and action.shape[1] == action_dim
    assert state.ndim == 2 and state.shape[1] == state_dim
    return state, action, replay_initial_positions


if __name__ == "__main__":
    load_zarr_episode(
        Path("/data/dexjoco_raw_datasets/dexjoco_raw_datasets/water_plant/water_plant_demo_5_2026-02-05_21-22-26_576352"),
        action_dim=22,
        state_dim=23
    )
