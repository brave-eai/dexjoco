"""Convert scalar-first action poses while keeping each arm with its hand."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def quaternion_to_rotvec(actions: NDArray, *, bimanual: bool) -> NDArray[np.float64]:
    """Convert quaternion actions to rotation-vector actions."""
    actions = np.asarray(actions, dtype=np.float64)
    blocks = []
    for offset in (0, 23) if bimanual else (0,):
        arm = actions[:, offset : offset + 23]
        rotation = Rotation.from_quat(arm[:, 3:7], scalar_first=True).as_rotvec()
        blocks.append(np.concatenate((arm[:, :3], rotation, arm[:, 7:]), axis=1))
    return np.concatenate(blocks, axis=1)


def rotvec_to_quaternion(actions: NDArray, *, bimanual: bool) -> NDArray[np.float64]:
    """Convert rotation-vector actions to quaternion actions."""
    actions = np.asarray(actions, dtype=np.float64)
    blocks = []
    for offset in (0, 22) if bimanual else (0,):
        arm = actions[:, offset : offset + 22]
        quaternion = Rotation.from_rotvec(arm[:, 3:6]).as_quat(scalar_first=True)
        blocks.append(np.concatenate((arm[:, :3], quaternion, arm[:, 6:]), axis=1))
    return np.concatenate(blocks, axis=1)
