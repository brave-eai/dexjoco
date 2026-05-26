import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Empty

import yaml

from ..process_protocol import ProgressBar
from ..utils import dict_to_slice, validate_data_paths
from .merge_episode_zarr import merge_episode_worker


def batch_merge_episode(
    output: Path,
    dataset_paths_cfg_path: Path,
    slice_cfg_path: Path,
    num_workers: int = 4,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
) -> None:
    """Merge multiple datasets into separate Zarr outputs in parallel.

    Args:
        output: Parent directory for per-dataset merged outputs.
        dataset_paths_cfg_path: YAML file containing dataset root paths.
        slice_cfg_path: YAML file containing per-dataset slice specifications.
        bad_episodes_cfg_path: YAML file containing per-dataset excluded episodes.
        skip_static_frames: Whether to remove static leading frames from each episode.
    """
    with open(dataset_paths_cfg_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    dataset_dirs = [Path(p) for p in dataset_paths]
    validate_data_paths(dataset_dirs, output)

    with open(slice_cfg_path, "r") as f:
        slice_config = yaml.safe_load(f)

    if bad_episodes_cfg_path is not None:
        with open(bad_episodes_cfg_path, "r") as f:
            bad_episodes_config = yaml.safe_load(f)
    else:
        bad_episodes_config = {}

    bars = {
        dataset_dir.name: ProgressBar(idx=idx, dataset_name=dataset_dir.name)
        for idx, dataset_dir in enumerate(dataset_dirs)
    }
    manager = mp.Manager()
    msg_queue = manager.Queue()
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for dataset_dir in dataset_dirs:
                future = executor.submit(
                    merge_episode_worker,
                    input_dir=dataset_dir,
                    output_dir=output / dataset_dir.name,
                    dataset_name=dataset_dir.name,
                    message_queue=msg_queue,
                    slice_cfg=dict_to_slice(slice_config.get(dataset_dir.name, {})),
                    bad_episodes=bad_episodes_config.get(dataset_dir.name),
                    skip_static_frames=skip_static_frames,
                )
                futures[dataset_dir.name] = future

            while True:
                try:
                    msg = msg_queue.get(timeout=1.0)
                except Empty:
                    if (
                        all(future.done() for future in futures.values())
                        and msg_queue.empty()
                    ):
                        break
                    else:
                        continue
                bars[msg.dataset_name].update(msg)

            for dataset_name, future in futures.items():
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker execution failed: {dataset_name}(error={e})")
    finally:
        manager.shutdown()
        for bar in bars.values():
            bar.close()

    failed = [name for name, bar in bars.items() if bar.state.status == "failed"]
    if failed:
        details = ", ".join(
            f"{name}(error={bars[name].state.last_error})" for name in failed
        )
        raise Exception(f"Failed datasets: {details}")

    for bar in bars.values():
        state = bar.state
        if state.done_msg is not None:
            print(state.done_msg)
