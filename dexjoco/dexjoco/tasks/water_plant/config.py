from typing import Literal

from ...sim.envs.panda_water_plant_env import PandaWaterPlantGymEnv
from ..config import TaskConfigBase
from ..obs_adapters import DexjocoObsAdapter
from ..sim_teleop import (
    SingleArmTeleopConfig,
    SingleArmViveHandTeleopWrapper,
)


class TaskConfig(TaskConfigBase):
    """Minimal water_plant config centered on simulated teleoperation capture."""

    proprio_keys = [
        "tcp_pose",
        "gripper_pose",
        "spray_ori_pose",
        "plant_ori_pose",
        "table_delta_height",
    ]
    teleop = SingleArmTeleopConfig(pose_scale=1.5)

    def get_environment(
        self,
        fake_env: bool = False,
        render_mode: Literal["rgb_array", "human"] = "human",
        randomize: bool = False,
        **kwargs,
    ):
        env = PandaWaterPlantGymEnv(
            render_mode=render_mode,
            randomize=randomize,
            hz=30,
            **kwargs,
        )
        if not fake_env:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
