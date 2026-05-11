import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.sim_teleop import BimanualTeleopConfig


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

        try:
            obs, _ = env.reset()

            # Optional image validation for headless evaluation.
            # images = {k: v for k, v in obs.items() if k != "state" and isinstance(v, np.ndarray)}
            # assert images, f"{env_name}: observation does not contain image arrays"

            for _ in range(100):
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    obs, _ = env.reset()

            print(f"{env_name}: ok")
        finally:
            env.close()


if __name__ == "__main__":
    main()
