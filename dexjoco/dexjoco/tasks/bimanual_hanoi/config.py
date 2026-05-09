from ..config import TaskConfigBase
from ..sim_teleop import (
    BimanualTeleopConfig,
    DualArmViveHandTeleopWrapper,
)
from ...sim.envs.panda_bimanual_hanoi_env import PandaBimanualHanoiGymEnv
from ..obs_adapters import DexjocoObsAdapter


class TaskConfig(TaskConfigBase):
    proprio_keys = ["tcp_pose", "gripper_pose", "hanoi_base_ori_pos", "table_delta_height"]
    teleop = BimanualTeleopConfig(pose_scale=1.5)

    def get_environment(self, fake_env=False, render_mode="human", randomize=False, **kwargs):
        env = PandaBimanualHanoiGymEnv(render_mode=render_mode, randomize=randomize, hz=30, **kwargs)
        if not fake_env:
            env = DualArmViveHandTeleopWrapper(env, self.teleop)
        env = DexjocoObsAdapter(env, proprio_keys=self.proprio_keys)
        return env

    def process_demos(self, demo):
        return demo
