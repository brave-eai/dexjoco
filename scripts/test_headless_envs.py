import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.sim_teleop import BimanualTeleopConfig

VIDEO_ROOT = Path("./test_headless_videos")
VIDEO_FPS = 30


def write_observation_videos(obs, writers):
    # DexjocoObsAdapter flattens proprioception into obs["state"] and exposes
    # image observations as top-level keys, so only those keys are recorded.
    for k, v in writers.items():
        if k not in obs:
            continue
        frame = obs[k]
        v.append_data(frame)


def zero_action(config):
    if isinstance(config.teleop, BimanualTeleopConfig):
        return np.zeros(46)
    return np.zeros(23)


def main():
    for env_name, config_cls in CONFIG_MAPPING.items():
        print(f"Testing environment: {env_name}")
        config = config_cls()
        env = config.get_environment(policy_mode=True, render_mode="rgb_array")
        action = zero_action(config)
        video_dir = VIDEO_ROOT / env_name
        video_dir.mkdir(parents=True, exist_ok=True)
        writers = {}

        try:
            obs, _ = env.reset()
            for k, v in obs.items():
                if not isinstance(v, np.ndarray):
                    continue  # not an image observation
                if v.ndim != 3:
                    continue  # not an image observation
                writers[k] = imageio.get_writer(video_dir / f"{k}.mp4", fps=VIDEO_FPS)
            write_observation_videos(obs, writers)

            for _ in range(100):
                obs, reward, terminated, truncated, info = env.step(action)
                write_observation_videos(obs, writers)
                if terminated or truncated:
                    obs, _ = env.reset()
                    write_observation_videos(obs, writers)

            print(f"{env_name}: ok")
        finally:
            for writer in writers.values():
                writer.close()
            env.close()


if __name__ == "__main__":
    main()
