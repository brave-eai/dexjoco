"""Evaluate PI 0.5 policy on DexJoCo simulation environments."""

import multiprocessing as mp
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from multiprocessing.synchronize import Event as MpEvent
from pathlib import Path
from queue import Empty
from typing import Literal

import imageio
import numpy as np
import torch
import yaml
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R

from .ipad_env import DexJoCoiPadEnv


@dataclass
class Observation:
    obs: dict
    timestamp: int


@dataclass
class Action:
    action: np.ndarray
    timestamp: int


ActionChunk = Action


def get_latest(q: mp.Queue):
    """Get the newest item from a queue and discard older buffered items."""
    latest = None
    try:
        while True:
            latest = q.get_nowait()
    except Empty:
        pass
    return latest


def _interp_rotvec_geodesic(
    rotvec0: np.ndarray, rotvec1: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate rotation vectors on SO(3) instead of component-wise lerp."""
    if t <= 0.0:
        return rotvec0.copy()
    if t >= 1.0:
        return rotvec1.copy()

    r0 = R.from_rotvec(rotvec0)
    r1 = R.from_rotvec(rotvec1)
    relative_rotvec = (r0.inv() * r1).as_rotvec()
    return (r0 * R.from_rotvec(relative_rotvec * t)).as_rotvec()


def _interp_single_arm_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate single-arm action [xyz, rotvec, hand]."""
    interp_action = (1.0 - t) * old_action + t * new_action
    rotvec_slice = slice(3, 6)
    interp_action[rotvec_slice] = _interp_rotvec_geodesic(
        old_action[rotvec_slice], new_action[rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def _interp_dual_arm_action(
    old_action: np.ndarray, new_action: np.ndarray, t: float
) -> np.ndarray:
    """Interpolate dual-arm action [r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand]."""
    interp_action = (1.0 - t) * old_action + t * new_action
    right_rotvec_slice = slice(3, 6)
    left_rotvec_slice = slice(25, 28)
    interp_action[right_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[right_rotvec_slice], new_action[right_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    interp_action[left_rotvec_slice] = _interp_rotvec_geodesic(
        old_action[left_rotvec_slice], new_action[left_rotvec_slice], t
    ).astype(interp_action.dtype, copy=False)
    return interp_action


def inference_process(
    obs_queue: mp.Queue,
    action_queue: mp.Queue,
    stop_event: MpEvent,
    port: int,
    inferencing_event: MpEvent,
    seed: int,
    host: str,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Inference worker: receive observations from queue and query policy server.

    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

    # Inference loop
    while not stop_event.is_set():
        obs: Observation | None = get_latest(obs_queue)
        if obs is None:
            stop_event.wait(0.01)
            continue
        # print("state", obs.obs["state"])
        # start_time = time.perf_counter()
        result = client.infer(obs.obs)
        # inference_duration = time.perf_counter() - start_time
        # print(f"Inference time: {inference_duration}s")
        action_chunk = result["actions"]

        action_timestamp = obs.timestamp
        action_queue.put(ActionChunk(action=action_chunk, timestamp=action_timestamp))

        inferencing_event.clear()


def receive_actions(
    action_queue: mp.Queue,
    actions_buffer: deque,
    now_timestamp: int,
    robot_type: Literal["single_arm", "dual_arm"],
):
    """Receive action chunks and merge them into a timestamped action buffer.

    now_timestamp has not been executed yet.
    """
    if robot_type == "single_arm":
        interp_action_fn = _interp_single_arm_action
    elif robot_type == "dual_arm":
        interp_action_fn = _interp_dual_arm_action
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}")

    # Drop expired actions that are older than the current timestamp.
    while actions_buffer and actions_buffer[0].timestamp < now_timestamp:
        actions_buffer.popleft()

    while True:
        try:
            action_chunk: ActionChunk = action_queue.get_nowait()

            # Chunk timestamp comes from observation, so it should not exceed now_timestamp.
            assert action_chunk.timestamp <= now_timestamp

            # All timestamp ranges below use half-open intervals: [start, end).
            action_chunk_timestamp_range = (
                now_timestamp,
                action_chunk.timestamp + action_chunk.action.shape[0],
            )
            if action_chunk_timestamp_range[1] <= now_timestamp:
                continue

            action = action_chunk.action[
                (action_chunk_timestamp_range[0] - action_chunk.timestamp) : (
                    action_chunk_timestamp_range[1] - action_chunk.timestamp
                )
            ]

            if actions_buffer:
                buffer_timestamp_range = (
                    actions_buffer[0].timestamp,
                    actions_buffer[-1].timestamp + 1,
                )
                assert buffer_timestamp_range[1] - buffer_timestamp_range[0] == len(
                    actions_buffer
                ), "Buffer timestamps must be continuous"
            else:
                buffer_timestamp_range = (now_timestamp, now_timestamp)

            # Blend overlapping actions already in buffer.
            overlap_range = (
                max(action_chunk_timestamp_range[0], buffer_timestamp_range[0]),
                min(action_chunk_timestamp_range[1], buffer_timestamp_range[1]),
            )
            overlap_len = overlap_range[1] - overlap_range[0]
            for ts in range(overlap_range[0], overlap_range[1]):
                buffer_idx = ts - buffer_timestamp_range[0]
                action_idx = ts - action_chunk_timestamp_range[0]

                # Keep interpolation away from 0/1 endpoints for smoother transitions.
                interp_t = (ts - overlap_range[0] + 1) / (overlap_len + 1)

                interp_action = interp_action_fn(
                    actions_buffer[buffer_idx].action,
                    action[action_idx],
                    interp_t,
                )
                actions_buffer[buffer_idx] = Action(action=interp_action, timestamp=ts)

            # Append non-overlapping tail actions.
            non_overlap_timestamp_range = (
                buffer_timestamp_range[1],
                action_chunk_timestamp_range[1],
            )
            for ts in range(
                non_overlap_timestamp_range[0], non_overlap_timestamp_range[1]
            ):
                non_overlap_action_idx = ts - action_chunk_timestamp_range[0]
                actions_buffer.append(
                    Action(action=action[non_overlap_action_idx], timestamp=ts)
                )
        except Empty:
            break


def main(
    config: Path,
    seed: int = 0,
    randomize: bool = False,
    randomize_dynamics: bool = False,
    port: int = 8000,
    host: str = "0.0.0.0",
    output: Path | None = None,
    replan_ratio: float = 0.8,  # Trigger replanning when remaining buffered actions are below this ratio.
    episodes: int = 50,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load config
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg["exp_name"]
    camera_mapping = cfg["camera_mapping"]
    robot_type = cfg["robot_type"]
    assert robot_type == "dual_arm"
    prompt = cfg["prompt"]
    action_horizon = 30

    # Setup output directory
    if output is None:
        output_dir = Path("outputs") / f"{config.stem}_seed{seed}"
    else:
        output_dir = output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = DexJoCoiPadEnv(
        camera_mapping=camera_mapping,
        seed=seed,
        randomize=randomize,
        randomize_dynamics=randomize_dynamics,
        prompt=prompt,
        exp_name=exp_name,
    )
    env.start()

    # Create queues and start inference process
    obs_queue = mp.Queue()
    action_queue = mp.Queue()  # Queue of action chunks from inference worker.
    stop_event = mp.Event()
    inferencing_event = mp.Event()

    inference_proc = mp.Process(
        target=inference_process,
        args=(obs_queue, action_queue, stop_event, port, inferencing_event, seed, host),
    )

    try:
        inference_proc.start()
        num_success = 0

        for ep in range(episodes):
            print(f"Episode {ep + 1}/{episodes}")

            # Setup video writers (use temp dir first, rename after episode)
            video_dir = output_dir / f"episode_{ep:03d}_temp"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_writers = {
                cam_name: imageio.get_writer(video_dir / f"{cam_name}.mp4", fps=30)
                for cam_name in camera_mapping.values()
            }

            # Reset environment
            env.reset()

            timestamp = 0
            actions_buffer = deque()  # Timestamp-ordered action buffer.

            if exp_name == "monitor_mousepad":
                for _ in range(30):
                    env.step(
                        action=np.array(
                            [
                                -4.4294e-01,
                                1.3729e-06,
                                1.5170e00,
                                -3.14156462e00,
                                -6.91584035e-05,
                                -1.40317984e-03,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0,
                                0.263,
                                0,
                                0,
                                0,
                            ]
                        )
                    )

            # Send initial observation
            obs_queue.put(Observation(env.get_obs(), timestamp))

            # Save first frame
            raw_images = env.get_raw_images()
            for cam_name, writer in video_writers.items():
                writer.append_data(raw_images[cam_name])

            in_stay_state = (
                False  # Track whether the previous step already used stay().
            )

            password = []  # Record pressed digit groups from the environment.
            first_digit_count = defaultdict(int)

            # Episode loop
            while True:
                receive_actions(
                    action_queue,
                    actions_buffer,
                    timestamp,
                    robot_type,
                )

                # Get action for current timestamp
                if actions_buffer:
                    assert actions_buffer[0].timestamp == timestamp, (
                        "Buffer head timestamp must match current timestamp"
                    )
                    action = actions_buffer.popleft().action
                    in_stay_state = False
                else:
                    # print(f"No action at timestamp {timestamp}, using stay")
                    pressed_digits = env.stay(continue_stay=in_stay_state)
                    if pressed_digits:
                        password.append(pressed_digits)
                    in_stay_state = True
                    timestamp += 1
                    raw_images = env.get_raw_images()
                    for cam_name, writer in video_writers.items():
                        writer.append_data(raw_images[cam_name])

                    # Send observation
                    if obs_queue.empty() and not inferencing_event.is_set():
                        inferencing_event.set()
                        obs_queue.put(Observation(env.get_obs(), timestamp))
                    continue

                # Execute action
                pressed_digits = env.step(action)
                if pressed_digits:
                    password.append(pressed_digits)
                timestamp += 1

                # Save frame
                raw_images = env.get_raw_images()
                for cam_name, writer in video_writers.items():
                    writer.append_data(raw_images[cam_name])

                # Send observation
                if (
                    len(actions_buffer) < replan_ratio * action_horizon
                    and obs_queue.empty()
                    and not inferencing_event.is_set()
                    and action_queue.empty()
                ):
                    inferencing_event.set()
                    obs_queue.put(Observation(env.get_obs(), timestamp))

                # Check termination
                if env.is_done:
                    if env.is_success:
                        num_success += 1
                        print("Success!")
                    else:
                        print("Failed")
                    break

            # Close video writers
            for writer in video_writers.values():
                writer.close()

            # Rename video_dir based on result
            result_suffix = "success" if env.is_success else "failure"
            password_suffix = []
            for digits in password:
                digits_str = "".join(str(d) for d in digits)
                password_suffix.append(digits_str)
            if password:
                password_suffix_str = "_".join(password_suffix)
            else:
                password_suffix_str = "no_password_input"
            final_video_dir = (
                output_dir / f"episode_{ep:03d}_{result_suffix}_{password_suffix_str}"
            )
            video_dir.rename(final_video_dir)

            if password:
                first_digit_count[password[0][0]] += 1
            else:
                first_digit_count["none"] += 1

            # Clear queues for next episode
            while not obs_queue.empty():
                time.sleep(0.1)
            while inferencing_event.is_set():
                time.sleep(0.1)
            while not action_queue.empty():
                action_queue.get()

        # Print final results
        print(
            f"\nSuccess rate: {num_success}/{episodes} ({100 * num_success / episodes:.1f}%)"
        )
        (output_dir / f"success_rate_{num_success}_{episodes}.txt").touch()

        with open(output_dir / "first_digit_count.yaml", "w") as f:
            yaml.dump(dict(first_digit_count), f)

    finally:
        stop_event.set()
        inference_proc.join(timeout=5)
        if inference_proc.is_alive():
            inference_proc.terminate()
        obs_queue.cancel_join_thread()
        obs_queue.close()
        action_queue.cancel_join_thread()
        action_queue.close()
        env.close()
