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
