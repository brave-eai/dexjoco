import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Empty

import yaml

from ..process_protocol import ProgressBar, WorkerMsg
from ..utils import dict_to_slice, validate_data_paths
from .merge_episode_lerobot import merge_episode_worker


def batch_merge_episode(
    output: Path,
    dataset_paths_cfg_path: Path,
    language_instruction_cfg_path: Path,
    selected_data_cfg_path: Path,
    slice_cfg_path: Path,
    num_workers: int = 4,
    bad_episodes_cfg_path: Path | None = None,
    skip_static_frames: bool = True,
    silent: bool = True,
) -> None:
    """Merge multiple datasets into separate LeRobot outputs in parallel.

    Args:
        output: Parent directory for per-dataset LeRobot outputs.
        dataset_paths_cfg_path: YAML file containing dataset root paths.
        language_instruction_cfg_path: YAML file containing per-dataset task
            instructions.
        selected_data_cfg_path: YAML file containing per-dataset selected action,
            state, and camera keys.
        slice_cfg_path: YAML file containing per-dataset slice specifications.
        num_workers: Maximum number of dataset workers running at the same time.
        bad_episodes_cfg_path: Optional YAML file containing per-dataset excluded
            episodes.
        skip_static_frames: Whether to remove static leading frames from each episode.
        silent: Whether to suppress stdout and stderr inside worker processes.

    Returns:
        None.

    Raises:
        AssertionError: If any configured dataset path does not exist.
        Exception: If output validation fails, any worker exits with an error, or
            any dataset reports a failed status.
    """
    with open(dataset_paths_cfg_path, "r") as f:
        dataset_paths = yaml.safe_load(f)
    dataset_dirs = [Path(p) for p in dataset_paths]
    validate_data_paths(dataset_dirs, output)

    # * load prompts
    with open(language_instruction_cfg_path, "r") as f:
        language_instruction_config = yaml.safe_load(f)

    # * load selected data config
    with open(selected_data_cfg_path, "r") as f:
        selected_data_config = yaml.safe_load(f)

    # * load bad episodes config
    if bad_episodes_cfg_path is not None:
        with open(bad_episodes_cfg_path, "r") as f:
            bad_episodes_config = yaml.safe_load(f)
    else:
        bad_episodes_config = {}

    # * load slice config
    with open(slice_cfg_path, "r") as f:
        slice_config = yaml.safe_load(f)

    # * start workers
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
                    language_instruction=language_instruction_config[dataset_dir.name],
                    slice_cfg=dict_to_slice(slice_config[dataset_dir.name]),
                    selected_data=selected_data_config[dataset_dir.name],
                    bad_episodes=bad_episodes_config.get(dataset_dir.name),
                    skip_static_frames=skip_static_frames,
                    silent=silent,
                )
                futures[dataset_dir.name] = future

            while True:
                try:
                    msg: WorkerMsg = msg_queue.get(timeout=1.0)
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

    # handle failure
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
