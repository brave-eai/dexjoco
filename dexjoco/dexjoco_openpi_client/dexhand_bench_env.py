"""
DexhandBenchEnv: Wrapper for simulation environment, adapted for PI 0.5 inference.

Usage:
    env = DexhandBenchEnv(
        exp_name="your_exp",
        camera_mapping={"camera_0": "ego", "camera_1": "wrist_right"},
    )
    env.start()
    env.reset()
    obs = env.get_obs()  # current single-frame processed observation
    env.step(action)     # action: [22] policy output format
    env.close()
"""

import sys

import copy

import numpy as np
from openpi_client import image_tools
from scipy.spatial.transform import Rotation as R


class DexhandBenchEnv:
    def __init__(
        self,
        exp_name: str,
        camera_mapping: dict[str, str],
        seed: int,
        randomize: bool,
        randomize_dynamics: bool,
        single_arm: bool,
        prompt: str,
        pad_state_dim46: bool = False,
    ):
        """
        Args:
            exp_name: Experiment name for CONFIG_MAPPING to get environment config
            camera_mapping: Camera name mapping {policy_key: env_key}
                e.g. {"camera_0": "ego", "camera_1": "wrist_right"}
        """
        self.exp_name = exp_name
        self.camera_mapping = camera_mapping
        self.seed = seed
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.single_arm = single_arm
        self.prompt = prompt
        self.pad_state_dim46 = pad_state_dim46

        self.env = None
        """
        keys:
            base, wrist, state, prompt: single arm
            base, wrist_left, wrist_right, state, prompt: dual arm
        """
        self.obs = {}  # the 1-frame observation used for policy input
        self._raw_obs: dict = {}  # Store latest raw images for video saving
        self._done = False
        self._success = False

    def start(self):
        """Start the simulation environment"""
        config = CONFIG_MAPPING[self.exp_name]()
        self.env = config.get_environment(
            seed=self.seed,  # type: ignore
            randomize=self.randomize,  # type: ignore
            randomize_dynamics=self.randomize_dynamics,  # type: ignore
        )

    def close(self):
        """Close the simulation environment"""
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self):
        """Reset environment and initialize current observation from reset output"""
        assert self.env is not None, "Environment not started. Call start() first."
        obs, _ = self.env.reset()
        self._done = False
        self._success = False

        self._update_raw_obs(obs)
        processed = self._process_obs(obs)
        self.obs = processed

    def step(self, action: np.ndarray):
        """
        Execute action

        Args:
            action: Policy output, shape 22/44
                [xyz(3), rotvec(3), hand(16)] / [xyz(3), rotvec(3), hand(16)]*2
        """
        assert self.env is not None, "Environment not started. Call start() first."
        env_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)

        self._done = bool(terminated)
        self._success = info.get("succeed", False)

        self._update_raw_obs(obs)
        self.obs = self._process_obs(obs)

    def get_obs(self) -> dict[str, np.ndarray]:
        return copy.deepcopy(self.obs)

    def stay(self, continue_stay: bool = False):
        """Keep current state by executing current state as action"""
        if continue_stay:
            # remain last stay state
            stay_state = self.last_stay_state
        else:
            stay_state = self.obs["state"]
            self.last_stay_state = stay_state
        if self.single_arm:
            # state: [arm(7), hand(16)]
            # action: [xyz(3), rotvec(3), hand(16)]
            arm = stay_state[:7]  # [xyz(3), quat(4)] but we need rotvec
            hand = stay_state[7:23]

            # Convert quat to rotvec for action format
            xyz = arm[:3]
            quat = arm[3:7]  # [w, x, y, z]
            rotvec = R.from_quat(quat, scalar_first=True).as_rotvec()

            action = np.concatenate([xyz, rotvec, hand])
        else:
            # state: [r_arm(7), l_arm(7), r_hand(16), l_hand(16)]
            # action: [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)]
            r_arm = stay_state[:7]  # [r_xyz(3), r_quat(4)] but we need rotvec
            l_arm = stay_state[7:14]
            r_hand = stay_state[14:30]
            l_hand = stay_state[30:46]

            # Convert quat to rotvec for action format
            r_xyz = r_arm[:3]
            r_quat = r_arm[3:7]  # [w, x, y, z]
            r_rotvec = R.from_quat(r_quat, scalar_first=True).as_rotvec()

            l_xyz = l_arm[:3]
            l_quat = l_arm[3:7]
            l_rotvec = R.from_quat(l_quat, scalar_first=True).as_rotvec()

            action = np.concatenate([r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand])
        self.step(action)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def _process_obs(self, env_obs: dict) -> dict[str, np.ndarray]:
        """
        Process environment observation to policy input format
        images are resized but state stays unnormalized
        """
        obs_dict = {}

        # Process images
        for policy_key, env_key in self.camera_mapping.items():
            img = env_obs[env_key][0]  # [H, W, C], uint8
            obs_dict[policy_key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            )

        # Process state
        if self.single_arm:
            state = env_obs["state"][0][:23]
            if self.pad_state_dim46:
                state = np.concatenate([state, np.zeros(46 - len(state))])
        else:
            state = env_obs["state"][0][:46]
        obs_dict["state"] = state

        obs_dict["prompt"] = self.prompt

        return obs_dict

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert policy output to environment action format
        """
        if self.single_arm:
            xyz = action[:3]
            rotvec = action[3:6]
            hand = action[6:22]

            quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)  # [w, x, y, z]

            env_action = np.concatenate([xyz, quat, hand])
        else:
            r_xyz = action[:3]
            r_rotvec = action[3:6]
            r_hand = action[6:22]
            l_xyz = action[22:25]
            l_rotvec = action[25:28]
            l_hand = action[28:44]

            r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)  # [w, x, y, z]
            l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)

            env_action = np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])

        return env_action

    def _update_raw_obs(self, env_obs: dict):
        """Store raw images for video saving"""
        self._raw_obs = {}
        for env_key in self.camera_mapping.values():
            self._raw_obs[env_key] = env_obs[env_key][0]  # [H, W, C], uint8

        # hack: zwz
        if "front" in env_obs.keys():
            self._raw_obs["front"] = env_obs["front"][0]

    def get_raw_images(self) -> dict[str, np.ndarray]:
        """Get raw images for video saving (original resolution)"""
        return self._raw_obs
