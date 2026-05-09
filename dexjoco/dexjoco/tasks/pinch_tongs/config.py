from ..config import TaskConfigBase
from ..sim_teleop import (
    SingleArmTeleopConfig,
    SingleArmViveHandTeleopWrapper,
)
from ...sim.envs.panda_pinch_tongs_env import PandaPinchTongsGymEnv
from ..obs_adapters import DexjocoObsAdapter


class TaskConfig(TaskConfigBase):
    proprio_keys = ["tcp_pose", "gripper_pose", "tongs_ori_pose", "table_delta_height"]
    teleop = SingleArmTeleopConfig(pose_scale=2.0)

    def get_environment(self, fake_env=False, render_mode="human", randomize=False, **kwargs):
        env = PandaPinchTongsGymEnv(render_mode=render_mode, randomize=randomize, hz=30, **kwargs)
        if not fake_env:
            env = SingleArmViveHandTeleopWrapper(env, self.teleop)
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
