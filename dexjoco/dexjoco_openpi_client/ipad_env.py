"""
DexJoCoiPadEnv: Wrapper for the DexJoCo iPad unlock simulation used by PI 0.5 inference.

Usage:
    env = DexJoCoiPadEnv(
        camera_mapping={"camera_0": "ego", "camera_1": "wrist_right"},
        seed=0,
        randomize=False,
        randomize_dynamics=False,
        prompt="unlock the ipad",
    )
    env.start()
    env.reset()
    obs = env.get_obs()  # current single-frame processed observation
    env.step(action)     # action: [44] dual-arm policy output
    env.close()
"""

import sys

sys.path.insert(0, "/home/brave/zwz/dexhand-bench/examples")

import copy

import numpy as np
from experiments.mappings import CONFIG_MAPPING
from openpi_client import image_tools
from scipy.spatial.transform import Rotation as R


class DexJoCoiPadEnv:
    def __init__(
        self,
        camera_mapping: dict[str, str],
        seed: int,
        randomize: bool,
        randomize_dynamics: bool,
        prompt: str,
        exp_name: str = "ipad_unlock",
    ):
        """
        Args:
            exp_name: Experiment name for CONFIG_MAPPING (default: ipad_unlock)
            camera_mapping: Camera name mapping {policy_key: env_key}
                e.g. {"camera_0": "ego", "camera_1": "wrist_right"}
            seed: Environment random seed
            randomize: Whether to randomize scene/object setup
            randomize_dynamics: Whether to randomize physics dynamics
            prompt: Language prompt passed to policy input
        """
        self.exp_name = exp_name
        self.camera_mapping = camera_mapping
        self.seed = seed
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.prompt = prompt

        self.env = None
        """
        Processed observation keys include:
            - image keys from camera_mapping (policy-side names)
            - state
            - prompt
        """
        self.obs = {}  # the 1-frame observation used for policy input
        self._raw_obs: dict = {}  # Store latest raw images for video saving
        self._done = False
        self._success = False

    def start(self):
        """Start the simulation environment"""
        assert self.exp_name == "ipad_unlock"
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
        """Reset environment and initialize the current processed observation."""
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
            action: Dual-arm policy output, shape [44]
                [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)]
        """
        assert self.env is not None, "Environment not started. Call start() first."
        env_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)

        self._done = bool(terminated)
        self._success = info.get("succeed", False)

        pressed_digits = info["pressed_digits"]

        self._update_raw_obs(obs)
        self.obs = self._process_obs(obs)

        return pressed_digits

    def get_obs(self) -> dict[str, np.ndarray]:
        """Get a deepcopy of the latest processed single-frame observation."""
        return copy.deepcopy(self.obs)

    def stay(self, continue_stay: bool = False):
        """Execute a hold action derived from current state and return pressed digits."""
        if continue_stay:
            # Reuse the previously cached hold state.
            stay_state = self.last_stay_state
        else:
            stay_state = self.obs["state"]
            self.last_stay_state = stay_state

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
        return self.step(action)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def _process_obs(self, env_obs: dict) -> dict[str, np.ndarray]:
        """
        Convert environment observation to policy input format.

        Images are resized/padded to 224x224 and kept as uint8.
        State keeps the first 46 dimensions without normalization.
        """
        obs_dict = {}

        # Process images
        for policy_key, env_key in self.camera_mapping.items():
            img = env_obs[env_key][0]  # [H, W, C], uint8
            obs_dict[policy_key] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, 224, 224)
            )

            # Process state
            state = env_obs["state"][0][:46]
        obs_dict["state"] = state

        obs_dict["prompt"] = self.prompt

        return obs_dict

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert policy action [44] to environment action [46].

        Input: [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)]
        Output: [r_xyz(3), r_quat(4), l_xyz(3), l_quat(4), r_hand(16), l_hand(16)]
        """
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

    def get_raw_images(self) -> dict[str, np.ndarray]:
        """Get raw images for video saving (original resolution)"""
        return self._raw_obs
