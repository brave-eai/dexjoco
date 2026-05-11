import numpy as np

from dexjoco.tasks.mappings import CONFIG_MAPPING
from dexjoco.tasks.sim_teleop import BimanualTeleopConfig


def zero_action(config):
    return np.zeros(46 if isinstance(config.teleop, BimanualTeleopConfig) else 23)


def main():
    for env_name, config_cls in CONFIG_MAPPING.items():
        print(f"Testing environment: {env_name}")
        config = config_cls()
        env = config.get_environment()
        action = zero_action(config)

        try:
            env.reset()
            for _ in range(100):
                _, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    env.reset()
            print(f"{env_name}: ok")
        finally:
            env.close()


if __name__ == "__main__":
    main()
