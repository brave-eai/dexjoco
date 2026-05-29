import numpy as np


def find_first_non_static_frame(action: np.ndarray) -> int:
    """Find the first frame where an episode starts changing.

    Args:
        action: Per-step action array ordered by time.

    Returns:
        Index of the first action whose value differs from the following action.

    Raises:
        ValueError: If all adjacent actions are identical.
    """
    for i in range(len(action) - 1):
        if not np.array_equal(action[i], action[i + 1]):
            return i
    raise ValueError("All actions in episode are identical (entirely static)")
