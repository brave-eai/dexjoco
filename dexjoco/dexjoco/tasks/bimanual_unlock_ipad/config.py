from typing import Literal

from ...sim.envs.panda_bimanual_unlock_ipad_env import PandaBimanualUnlockIpadGymEnv
from ..config import TaskConfigBase
from ..obs_adapters import DexjocoObsAdapter
from ..policy_wrappers import DualArmPolicyWrapper
from ..sim_teleop import (
    BimanualTeleopConfig,
    DualArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "stand_ori_pose",
        "ipad_ori_pose",
        "table_delta_height",
    ]
    teleop = BimanualTeleopConfig(pose_scale=1.5)

    def get_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=False,
        **kwargs,
    ):
        env = PandaBimanualUnlockIpadGymEnv(
            render_mode=render_mode, randomize=randomize, hz=30, **kwargs
        )
        if policy_mode:
            env = DualArmPolicyWrapper(env)
        else:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
