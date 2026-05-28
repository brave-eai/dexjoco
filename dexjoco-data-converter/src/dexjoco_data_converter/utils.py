import multiprocessing as mp
from pathlib import Path

import numpy as np


def normalize_array_shape(arr: np.ndarray) -> np.ndarray:
    """Normalize per-step arrays before conversion.

    Args:
        arr: Source array from an episode replay buffer.

    Returns:
        Array with a singleton second dimension removed when the input shape is
        ``(T, 1, D)``; otherwise, the original array.
    """
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr.squeeze(axis=1)
    return arr


def pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Right-pad the feature dimension of a 2D array with zeros.

    Args:
        arr: Array whose second dimension is the feature dimension.
        target_dim: Required feature dimension.

    Returns:
        Array with the same leading dimension and a second dimension equal to
        ``target_dim``.
    """
    assert arr.ndim == 2, f"arr must have 2 dims (t, n), got {arr.shape}"
    current_dim = arr.shape[1]
    if current_dim > target_dim:
        raise Exception(f"arr dim {current_dim} exceeds target dim {target_dim}")
    if current_dim == target_dim:
        return arr

    pad_width = [(0, 0)] * arr.ndim
    pad_width[1] = (0, target_dim - current_dim)
    return np.pad(arr, pad_width, mode="constant", constant_values=0)


def dict_to_slice(d: dict[str, list]):
    """Convert serialized slice specifications to Python slice objects.

    Args:
        d: Mapping from data key to a list of slice arguments accepted by
            ``slice(*value)``.

    Returns:
        Mapping from data key to the corresponding ``slice`` object.
    """
    res = {}
    for k, v in d.items():
        res[k] = slice(*v)
    return res


def validate_data_paths(datasets_path: list[Path], output: Path):
    """Validate dataset roots and prepare the batch output directory.

    Args:
        datasets_path: Dataset root directories to process.
        output: Parent output directory that contains one subdirectory per
            dataset.

    Returns:
        None.

    Raises:
        AssertionError: If any dataset path does not exist.
        Exception: If a target dataset output directory already contains files.
    """
    assert all(p.exists() for p in datasets_path), "Some datasets path do not exist"

    output.mkdir(parents=True, exist_ok=True)
    for dataset_dir in datasets_path:
        dataset_output = output / dataset_dir.name
        if dataset_output.exists() and any(dataset_output.iterdir()):
            raise Exception(f"Output directory is not empty: {dataset_output}")


def terminate_process(
    process: mp.Process, process_name: str | None = None, timeout: float = 10.0
) -> None:
    """Wait for a process and terminate it if it does not exit in time.

    Args:
        process: Process to clean up.
        process_name: Name used in worker failure messages.
        timeout: Seconds to wait before terminating the process.
    """
    process.join(timeout=timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=timeout)

    if process.exitcode != 0:
        name = process_name or process.name
        print(f"Worker execution failed: {name}(exitcode={process.exitcode})")
