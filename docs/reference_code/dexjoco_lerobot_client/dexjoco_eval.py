import threading
import time
from pathlib import Path
from queue import Queue
from typing import Literal

import imageio
import numpy as np
import yaml
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.transport import services_pb2

from .async_observation_robot_client import AsyncObservationRobotClient
from .config_dexjoco_robot import DexJoCoRobotConfig
from .dexjoco_robot import DexJoCoRobot


def reset_client_runtime_state(client: AsyncObservationRobotClient) -> None:
    with client.action_queue_lock:
        client.action_queue = Queue()
    client.clear_pending_observations()
    with client.latest_action_lock:
        client.latest_action = -1
    client.action_chunk_size = -1
    client.must_go.set()
    client.fps_tracker.reset()


def eval_n_episodes(
    client: AsyncObservationRobotClient,
    n_episodes: int,
    task: str,
    video_out_root: Path,
) -> float:
    assert isinstance(client.robot, DexJoCoRobot), (
        "client.robot must be an instance of DexJoCoRobot"
    )

    video_keys = [
        k
        for k in client.robot.observation_features
        if isinstance(client.robot.observation_features[k], tuple)
    ]

    successes = []
    for episode in range(n_episodes):
        video_writers = {}
        episode_success = False
        episode_out_path = video_out_root / f"{episode}_temp"
        episode_out_path.mkdir(exist_ok=True, parents=True)
        video_writers = {
            k: imageio.get_writer(episode_out_path / f"{k}.mp4", fps=30)
            for k in video_keys
        }

        try:
            # reset server state
            client.wait_for_all_observations_sent()  # prevent observations from last episode being sent after reset
            reset_client_runtime_state(client)
            client.stub.Ready(services_pb2.Empty())  # type: ignore
            client.robot.reset()

            # Legacy alignment hook retained only as a reference example.
            if client.robot.exp_name == "click_mouse":
                for _ in range(30):
                    obs, *_ = client.robot.env.step(
                        action=np.array([
                            -4.4294e-01,
                            1.3729e-06,
                            1.5170e00,
                            1.3860e-05,
                            -1.0000e00,
                            -2.2014e-05,
                            -4.4665e-04,
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
                        ])
                    )
                    client.robot.observation = client.robot._process_observation(obs)

            in_stay_state = False

            episode_start_time = time.time()

            # send the initial observation
            client.must_go.set()
            client.control_loop_observation(task=task)

            while True:
                # check if there is legal action
                if client.actions_available():
                    with client.action_queue_lock:
                        action = client.action_queue.queue[0]
                    if action.get_timestamp() >= episode_start_time:
                        action_legal = True
                    else:
                        # drop the action from last episode
                        with client.action_queue_lock:
                            client.action_queue.get_nowait()
                        time.sleep(client.config.environment_dt)
                        continue
                else:
                    action_legal = False

                # if legal action, execute it; otherwise, stay and wait
                if action_legal:
                    client.control_loop_action()
                    in_stay_state = False
                else:
                    client.robot.stay(in_stay_state)
                    in_stay_state = True
                    time.sleep(client.config.environment_dt)

                if client._ready_to_send_observation():
                    client.must_go.set()
                    client.control_loop_observation(task=task)
                    # print("timestamp: ", client.latest_action)

                # save video
                obs = client.robot.get_observation()
                for k in video_keys:
                    video_writers[k].append_data(obs[k])

                if client.robot.is_done:
                    break

            successes.append(client.robot.is_success)
            episode_success = client.robot.is_success
        finally:
            for writer in video_writers.values():
                writer.close()
        episode_out_path.rename(
            video_out_root / f"{episode}_{'success' if episode_success else 'failure'}"
        )

    (video_out_root / f"success_rate_{sum(successes) / len(successes):.3f}.txt").touch()

    return sum(successes) / len(successes)


def main(
    env_name: str,
    config_path: Path,
    pretrained_path: Path,
    policy_type: Literal["act", "groot"],
    seed: int,
    randomize: bool,
    randomize_dynamics: bool = False,
    output_root: Path = Path("./outputs"),
    server_address: str = "127.0.0.1:8080",
    n_episodes: int = 50,
    replan_threshold: float = 0.8,
    pad_state_dim46: bool = False,
) -> None:
    video_out_root = (
        output_root
        / f"{config_path.stem}_{policy_type}_seed{seed}_{'rand' if randomize else 'norand'}"
    )

    if video_out_root.exists():
        raise FileExistsError(
            f"Output path {video_out_root} already exists. Please remove it first."
        )

    robot_cfg = DexJoCoRobotConfig(
        id=env_name,
        config_path=config_path,
        seed=seed,
        randomize=randomize,
        randomize_dynamics=randomize_dynamics,
        pad_state_dim46=pad_state_dim46,
    )

    if policy_type == "act":
        actions_per_chunk = 100
    elif policy_type == "groot":
        actions_per_chunk = 16
    else:
        raise ValueError(f"Invalid policy_type: {policy_type}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        task = cfg["task"]

    robot_client_cfg = RobotClientConfig(
        policy_type=policy_type,
        pretrained_name_or_path=str(pretrained_path),
        robot=robot_cfg,
        actions_per_chunk=actions_per_chunk,
        task=task,
        server_address=server_address,
        policy_device="cuda",
        client_device="cpu",
        fps=30,
        aggregate_fn_name="latest_only",  # action rotvec does not support averaging
        chunk_size_threshold=replan_threshold,
    )

    client = AsyncObservationRobotClient(robot_client_cfg)
    client.start()

    action_receiver_thread = threading.Thread(
        target=client.receive_actions, daemon=True
    )
    action_receiver_thread.start()

    # wait for receive_actions to be ready
    client.start_barrier.wait()

    try:
        success_rate = eval_n_episodes(
            client, n_episodes=n_episodes, task=task, video_out_root=video_out_root
        )
        print(f"success_rate={success_rate:.3f}")
    finally:
        client.stop()
        action_receiver_thread.join()
