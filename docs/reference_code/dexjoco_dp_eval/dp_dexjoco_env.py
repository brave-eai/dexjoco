"""
DexJoCoEnv: Unified wrapper for single-arm and dual-arm simulation environments
adapted for Diffusion Policy inference.

Usage:
    env = DexJoCoEnv(
        exp_name="your_exp",
        image_size=(240, 240),
        camera_mapping={"camera_0": "ego", "camera_1": "wrist_right"},
        seed=0,
        randomize=False,
        single_arm=True,
    )
    env.start()
    env.reset()
    obs = env.get_obs()  # {key: [2, ...]} two-frame history
    env.step(action)     # action: [22] for single-arm, [44] for dual-arm
    env.close()
"""

from collections import deque

import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform
from dexjoco.tasks import CONFIG_MAPPING
from scipy.spatial.transform import Rotation as R


class DexJoCoEnv:
    def __init__(
        self,
        exp_name: str,
        image_size: tuple[int, int],
        camera_mapping: dict[str, str],
        seed: int,
        randomize: bool,
        randomize_dynamics: bool,
        single_arm: bool,
        pad_state_dim46: bool = False,
    ):
        """Create a reference wrapper around a DexJoCo task environment."""
        self.exp_name = exp_name
        self.image_size = image_size
        self.camera_mapping = camera_mapping
        self.seed = seed
        self.randomize = randomize
        self.randomize_dynamics = randomize_dynamics
        self.single_arm: bool = single_arm
        self.pad_state_dim46 = pad_state_dim46

        self.env = None
        self.obs_queue: deque = deque(maxlen=2)
        self._raw_obs: dict = {}  # Store latest raw images for video saving
        self._done = False
        self._success = False

        # Keep the resize explicit so the reference matches the policy wrapper shape.
        self._img_transform = get_image_transform(
            input_res=(640, 640),
            output_res=image_size,
            bgr_to_rgb=False,
        )

    def start(self):
        """Start the simulation environment."""
        config = CONFIG_MAPPING[self.exp_name]()
        self.env = config.get_environment(
            seed=self.seed,  # type: ignore
            randomize=self.randomize,  # type: ignore
            randomize_dynamics=self.randomize_dynamics,  # type: ignore
        )

    def close(self):
        """Close the simulation environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def reset(self):
        """Reset the environment and seed the history queue with the first frame."""
        assert self.env is not None, "Environment not started. Call start() first."
        obs, _ = self.env.reset()
        self._done = False
        self._success = False
        self._update_raw_obs(obs)
        processed = self._process_obs(obs)
        self.obs_queue.clear()
        # Fill two-frame history with first frame
        self.obs_queue.append(processed)
        self.obs_queue.append(processed)

    def step(self, action: np.ndarray):
        """Execute one policy action in environment format.

        Single-arm actions are 22-D `[xyz, rotvec, hand]`.
        Dual-arm actions are 44-D `[r_xyz, r_rotvec, r_hand, l_xyz, l_rotvec, l_hand]`.
        """
        assert self.env is not None, "Environment not started. Call start() first."
        env_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self.env.step(env_action)

        self._done = bool(terminated)
        self._success = info.get("succeed", False)

        self._update_raw_obs(obs)
        processed = self._process_obs(obs)
        self.obs_queue.append(processed)

    def get_obs(self) -> dict[str, np.ndarray]:
        """Return a two-frame history to match the Diffusion Policy reference."""
        obs_dict = {}
        keys = list(self.obs_queue[0].keys())
        for key in keys:
            obs_dict[key] = np.stack([self.obs_queue[i][key] for i in range(2)], axis=0)
        return obs_dict

    def stay(self, continue_stay: bool):
        """Keep the current pose by replaying the latest state as an action."""
        if continue_stay:
            assert hasattr(self, "last_stay_action"), (
                "No previous stay action found. Call stay(False) first to initialize."
            )
            stay_action = self.last_stay_action
        else:
            current_state = self.obs_queue[-1]["state"]
            if self.single_arm:
                # State layout: [arm(7), hand(16)].
                # Action layout: [xyz(3), rotvec(3), hand(16)].
                arm = current_state[:7]  # [xyz(3), quat(4)] but we need rotvec
                hand = current_state[7:23]

                # Convert quaternion state back to rotvec action parameters.
                xyz = arm[:3]
                quat = arm[3:7]  # [w, x, y, z]
                rotvec = R.from_quat(quat, scalar_first=True).as_rotvec()

                stay_action = np.concatenate([xyz, rotvec, hand])
            else:
                # State layout: [r_arm(7), l_arm(7), r_hand(16), l_hand(16)].
                # Action layout: [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)].
                r_arm = current_state[:7]  # [r_xyz(3), r_quat(4)] but we need rotvec
                l_arm = current_state[7:14]
                r_hand = current_state[14:30]
                l_hand = current_state[30:46]

                # Convert quaternion state back to rotvec action parameters.
                r_xyz = r_arm[:3]
                r_quat = r_arm[3:7]  # [w, x, y, z]
                r_rotvec = R.from_quat(r_quat, scalar_first=True).as_rotvec()

                l_xyz = l_arm[:3]
                l_quat = l_arm[3:7]
                l_rotvec = R.from_quat(l_quat, scalar_first=True).as_rotvec()

                stay_action = np.concatenate([
                    r_xyz,
                    r_rotvec,
                    r_hand,
                    l_xyz,
                    l_rotvec,
                    l_hand,
                ])
            self.last_stay_action = stay_action
        self.step(stay_action)

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def is_success(self) -> bool:
        return self._success

    def _process_obs(self, env_obs: dict) -> dict[str, np.ndarray]:
        """Process environment observations into the policy input format."""
        obs_dict = {}

        for policy_key, env_key in self.camera_mapping.items():
            img = env_obs[env_key]  # [H, W, C], uint8
            img = self._img_transform(img)
            img = img.astype(np.float32) / 255.0
            obs_dict[policy_key] = np.moveaxis(img, -1, 0)  # [C, H, W]

        state = env_obs["state"]
        if self.single_arm:
            arm = state[:7]
            hand = state[7:23]
            obs_dict["state"] = np.concatenate([arm, hand]).astype(np.float32)
            if self.pad_state_dim46:
                # Pad to 46 dims for the shared model interface.
                obs_dict["state"] = np.concatenate([
                    obs_dict["state"],
                    np.zeros(46 - 23, dtype=np.float32),
                ])
        else:
            r_arm = state[:7]
            l_arm = state[7:14]
            r_hand = state[14:30]
            l_hand = state[30:46]
            obs_dict["state"] = np.concatenate([r_arm, l_arm, r_hand, l_hand]).astype(
                np.float32
            )

        return obs_dict

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert policy output to environment action format.

        Single-arm:
            Input: [xyz(3), rotvec(3), hand(16)] = 22
            Output: [xyz(3), quat(4), hand(16)] = 23

        Dual-arm:
            Input: [r_xyz(3), r_rotvec(3), r_hand(16), l_xyz(3), l_rotvec(3), l_hand(16)] = 44
            Output: [r_xyz(3), r_quat(4), l_xyz(3), l_quat(4), r_hand(16), l_hand(16)] = 46
        """
        if self.single_arm:
            xyz = action[:3]
            rotvec = action[3:6]
            hand = action[6:22]
            quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)
            processed_action = np.concatenate([xyz, quat, hand])
        else:
            r_xyz = action[:3]
            r_rotvec = action[3:6]
            r_hand = action[6:22]
            l_xyz = action[22:25]
            l_rotvec = action[25:28]
            l_hand = action[28:44]
            r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)
            processed_action = np.concatenate([
                r_xyz,
                r_quat,
                l_xyz,
                l_quat,
                r_hand,
                l_hand,
            ])
        return processed_action

    def _update_raw_obs(self, env_obs: dict):
        """Store raw images for video saving"""
        self._raw_obs = {}
        for env_key in self.camera_mapping.values():
            self._raw_obs[env_key] = env_obs[env_key][0]  # [H, W, C], uint8

    def get_raw_images(self) -> dict[str, np.ndarray]:
        """Get raw images for video saving (original resolution)"""
        return self._raw_obs
