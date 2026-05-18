# Reference implementation for adapting DexJoCo to the LeRobot robot interface.

import logging
from functools import cached_property

import numpy as np
import yaml
from dexjoco.tasks import CONFIG_MAPPING
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from scipy.spatial.transform import Rotation as R
from typing_extensions import override

from .config_dexjoco_robot import DexJoCoRobotConfig


class DexJoCoRobot(Robot):
    config_class = DexJoCoRobotConfig
    name = "dexjoco_robot"

    def __init__(self, config: DexJoCoRobotConfig):
        # * We don't call super().__init__() because we don't need calibration, robot_type and name
        # super().__init__(config)
        self.config = config
        assert config.id is not None, (
            "RobotConfig.id must be specified (as exp_name for CONFIG_MAPPING) for DexJoCo"
        )
        self.exp_name = config.id

        with open(config.config_path, "r") as f:
            cfg = yaml.safe_load(f)
            self.observation_features_cfg = cfg["observation_features"]
            self.action_features_cfg = cfg["action_features"]
            self.single_arm = cfg["single_arm"]
            self.model_env_image_map = cfg.get("model_env_image_map")

        self.done = False
        self.success = False

        self._is_connected = False

        self.seed = config.seed
        self.randomize = config.randomize
        self.randomize_dynamics = config.randomize_dynamics

    @check_if_not_connected
    def reset(self) -> None:
        """Reset the robot to its initial state."""
        obs, _ = self.env.reset()
        self.observation = self._process_observation(obs)
        self.done = False
        self.success = False

    @check_if_already_connected
    @override
    def connect(self, calibrate: bool = True) -> None:
        _ = calibrate  # Not used

        config = CONFIG_MAPPING[self.exp_name]()
        self.env = config.get_environment(
            seed=self.seed,  # type: ignore
            randomize=self.randomize,  # type: ignore
            randomize_dynamics=self.randomize_dynamics,  # type: ignore
        )

        self._is_connected = True
        logging.info(f"{self.exp_name} simulator created")

    @check_if_not_connected
    @override
    def disconnect(self) -> None:
        self.env.close()
        self._is_connected = False
        logging.info(f"{self.exp_name} simulator closed")

    @cached_property
    @override
    def observation_features(self) -> dict[str, type | tuple]:
        """Define the observation space for dataset.

        ref: lerobot/datasets/utils.py::hw_to_dataset_features
        Only float(state) and tuple(image) is supported now, PolicyFeature is not supported yet.
        """

        # build state features
        features = {}
        for item in self.observation_features_cfg["state"]:
            if isinstance(item, list):
                for name in item:
                    features[name] = float
            elif isinstance(item, dict):
                assert len(item) == 1, (
                    "Only one state feature is allowed in each dict item"
                )
                name, length = next(iter(item.items()))
                features.update({f"{name}_{i}": float for i in range(length)})
            else:
                raise ValueError("Invalid observation_features config format")

        # build image features
        for name, shape in self.observation_features_cfg["images"].items():
            features[name] = tuple(shape)

        return features

    @cached_property
    @override
    def action_features(self) -> dict[str, type]:
        """Define the action space.
        Must in the same order as model output, type must be float
        """
        features = {}
        for name in self.action_features_cfg:
            if isinstance(name, list):
                for n in name:
                    features[n] = float
            elif isinstance(name, dict):
                assert len(name) == 1, (
                    "Only one action feature is allowed in each dict item"
                )
                name, length = next(iter(name.items()))
                features.update({f"{name}_{i}": float for i in range(length)})
            else:
                raise ValueError("Invalid action_features config format")

        return features

    @check_if_not_connected
    @override
    def get_observation(self) -> RobotObservation:
        # Must be exactly the same as observation_features
        return self.observation

    @check_if_not_connected
    @override
    def send_action(self, action: RobotAction) -> RobotAction:
        # Must be exactly the same as action_features
        action_array = np.array([float(action[k]) for k in self.action_features.keys()])

        if self.single_arm:
            xyz = action_array[:3]
            rot_vec = action_array[3:6]
            hand = action_array[6:]
            quat = R.from_rotvec(rot_vec).as_quat(scalar_first=True)

            action_array = np.concatenate([xyz, quat, hand])
        else:
            r_xyz = action_array[:3]
            r_rot_vec = action_array[3:6]
            r_hand = action_array[6:22]
            l_xyz = action_array[22:25]
            l_rot_vec = action_array[25:28]
            l_hand = action_array[28:44]

            r_quat = R.from_rotvec(r_rot_vec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rot_vec).as_quat(scalar_first=True)

            action_array = np.concatenate([
                r_xyz,
                r_quat,
                l_xyz,
                l_quat,
                r_hand,
                l_hand,
            ])

        obs, reward, terminated, truncated, info = self.env.step(action_array)

        self.observation = self._process_observation(obs)
        self.done = bool(terminated)
        self.success = info["succeed"]

        return action

    def stay(self, continue_stay: bool):
        # Keep the simulator alive by replaying the current pose.
        if continue_stay:
            assert hasattr(self, "last_stay_action"), (
                "last_stay_action not found, cannot continue stay"
            )
            stay_action = self.last_stay_action
        else:
            action_list = []
            for key in self.observation_features.keys():
                # all float in state is the stay action
                if self.observation_features[key] is float:
                    action_list.append(float(self.observation[key]))
            stay_action = np.array(action_list)
            self.last_stay_action = stay_action

        obs, reward, terminated, truncated, info = self.env.step(stay_action)
        self.observation = self._process_observation(obs)
        self.done = bool(terminated)
        self.success = info["succeed"]

    @property
    def is_done(self) -> bool:
        return self.done

    @property
    def is_success(self) -> bool:
        return self.success

    def _process_observation(self, obs):
        if self.single_arm:
            arm_state = obs["state"][:7]
            hand_state = obs["state"][7:23]
            all_state = np.concatenate([arm_state, hand_state])
            if self.config.pad_state_dim46:
                all_state = np.concatenate([all_state, np.zeros(46 - len(all_state))])
        else:
            r_arm_state = obs["state"][:7]
            l_arm_state = obs["state"][7:14]
            arm_state = np.concatenate([r_arm_state, l_arm_state])
            r_hand_state = obs["state"][14:30]
            l_hand_state = obs["state"][30:46]
            hand_state = np.concatenate([r_hand_state, l_hand_state])
            all_state = np.concatenate([arm_state, hand_state])

        observations = {}
        state_idx = 0
        for name, dtype in self.observation_features.items():
            if dtype is not float:
                continue
            observations[name] = float(all_state[state_idx])
            state_idx += 1
        assert state_idx == len(all_state)

        # Map model image keys to environment image keys.
        for model_img_name in self.observation_features_cfg["images"]:
            if self.model_env_image_map is not None:
                env_image_name = self.model_env_image_map[model_img_name]
            else:
                env_image_name = model_img_name
            observations[model_img_name] = obs[env_image_name][0]
        return observations

    @property
    @override
    def is_connected(self) -> bool:
        return self._is_connected

    @override
    def calibrate(self) -> None:
        """Calibration not needed."""
        logging.info("Calibration not required")

    @property
    @override
    def is_calibrated(self) -> bool:
        return True

    @override
    def configure(self) -> None:
        """Configure robot (no-op for sim robot)."""
        pass
