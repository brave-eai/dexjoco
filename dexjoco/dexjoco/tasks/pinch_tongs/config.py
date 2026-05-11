from typing import Literal

from ...sim.envs.panda_pinch_tongs_env import PandaPinchTongsGymEnv
from ..config import TaskConfigBase
from ..obs_adapters import DexjocoObsAdapter
from ..policy_wrappers import SingleArmPolicyWrapper
from ..sim_teleop import (
    SingleArmTeleopConfig,
    SingleArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    proprio_keys = ["tcp_pose", "gripper_pose", "tongs_ori_pose", "table_delta_height"]
    teleop = SingleArmTeleopConfig(pose_scale=2.0)

    def get_environment(
        self,
        policy_mode=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=False,
        **kwargs,
    ):
        env = PandaPinchTongsGymEnv(
            render_mode=render_mode, randomize=randomize, hz=30, **kwargs
        )
        if policy_mode:
            env = SingleArmPolicyWrapper(env)
        else:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
