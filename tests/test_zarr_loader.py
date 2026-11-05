"""Check raw robot layout conversion without changing scene restoration values."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from dexjoco_data_converter.zarr_loader import load_zarr_episode


@pytest.mark.parametrize("bimanual", [False, True])
def test_raw_state_layout(tmp_path: Path, bimanual: bool) -> None:
    state_dim = 46 if bimanual else 23
    state = np.arange(2 * (state_dim + 32), dtype=np.float64).reshape(2, 1, state_dim + 32)
    action = np.zeros((2, state_dim), dtype=np.float64)
    action[:, 3] = 1.0
    if bimanual:
        action[:, 26] = 1.0
    root = zarr.open_group(str(tmp_path / "replay.zarr"), mode="w")
    data = root.create_group("data")
    data.create_dataset("state", data=state)
    data.create_dataset("action", data=action)

    converted, quaternion, rotvec, scene = load_zarr_episode(tmp_path, state_dim, state_dim)
    if bimanual:
        np.testing.assert_array_equal(converted[:, :7], state[:, 0, :7])
        np.testing.assert_array_equal(converted[:, 7:23], state[:, 0, 14:30])
        np.testing.assert_array_equal(converted[:, 23:30], state[:, 0, 7:14])
        np.testing.assert_array_equal(converted[:, 30:46], state[:, 0, 30:46])
    else:
        np.testing.assert_array_equal(converted, state[:, 0, :state_dim])
    np.testing.assert_array_equal(scene, state[0, 0, state_dim:])
    np.testing.assert_array_equal(quaternion, action)
    assert rotvec.shape == (2, 44 if bimanual else 22)
    np.testing.assert_array_equal(root["data/state"][:], state)
