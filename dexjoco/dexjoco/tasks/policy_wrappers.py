import gymnasium as gym
import numpy as np


class SingleArmPolicyWrapper(gym.ActionWrapper):
    """Adapt single-arm policy actions to the raw single-arm task format.

    Policy action layout:
      - shape: (23,)
      - action[0:3]: end-effector target position, xyz
      - action[3:7]: end-effector target quaternion, wxyz
      - action[7:23]: Allegro hand joint targets, 16 values

    Raw single-arm tasks consume the same 23-dimensional array:
      - [target_position_3, target_quaternion_4, allegro_joints_16]
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (23,):
            raise ValueError(
                f"Expected single-arm policy action shape (23,), got {action.shape}."
            )
        return action


class DualArmPolicyWrapper(gym.ActionWrapper):
    """Adapt bimanual policy actions to the raw bimanual task format.

    Policy action layout:
      - shape: (46,)
      - action[0:7]: right arm target pose, [x, y, z, qw, qx, qy, qz]
      - action[7:14]: left arm target pose, [x, y, z, qw, qx, qy, qz]
      - action[14:30]: right Allegro hand joint targets, 16 values
      - action[30:46]: left Allegro hand joint targets, 16 values

    Raw bimanual tasks consume a dictionary:
      - "right": shape (23,), [right_pose_7, right_hand_joints_16]
      - "left": shape (23,), [left_pose_7, left_hand_joints_16]
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(46,),
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (46,):
            raise ValueError(
                f"Expected bimanual policy action shape (46,), got {action.shape}."
            )

        right_pose = action[0:7]
        left_pose = action[7:14]
        right_hand = action[14:30]
        left_hand = action[30:46]

        return {
            "right": np.concatenate([right_pose, right_hand], axis=0),
            "left": np.concatenate([left_pose, left_hand], axis=0),
        }
