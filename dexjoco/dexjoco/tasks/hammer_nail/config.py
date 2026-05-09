from typing import Literal

from ...sim.envs.panda_hammer_nail_env import PandaHammerNailGymEnv
from ..config import TaskConfigBase
from ..obs_adapters import DexjocoObsAdapter
from ..sim_teleop import (
    SingleArmTeleopConfig,
    SingleArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "hammer_ori_pose",
        "nail_ori_pose",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig(pose_scale=1.5)

    def get_environment(
        self,
        fake_env=False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize=False,
        **kwargs,
    ):
        env = PandaHammerNailGymEnv(
            render_mode=render_mode, randomize=randomize, hz=30, **kwargs
        )
        if not fake_env:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
