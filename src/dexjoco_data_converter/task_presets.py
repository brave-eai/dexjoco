"""Conversion presets matching dexjoco/dexjoco/tasks/*/config.py.

Prompts match configs/rand_obj/*.yaml.

Replay fields follow proprio_keys after tcp_pose and gripper_pose. Sizes
come from the corresponding simulation environments: poses have 7 values
(position + quaternion), positions have 3, and table height has 1.
"""

from dataclasses import dataclass
from typing import Literal

TaskName = Literal[
    "water_plant",
    "click_mouse",
    "pinch_tongs",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
]


@dataclass(frozen=True)
class TaskPreset:
    is_bimanual: bool
    prompt: str
    replay_initial_position_fields: tuple[tuple[str, int], ...]


TASK_PRESETS: dict[TaskName, TaskPreset] = {
    "water_plant": TaskPreset(
        is_bimanual=False,
        prompt="Grasp the watering can and apply water to the plant.",
        replay_initial_position_fields=(
            ("spray_ori_pose", 7),
            ("plant_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "click_mouse": TaskPreset(
        is_bimanual=False,
        prompt="Move the mouse to the purple mouse pad and click the left mouse button.",
        replay_initial_position_fields=(
            ("mouse_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "pinch_tongs": TaskPreset(
        is_bimanual=False,
        prompt="Grasp the tongs and perform three consecutive open-close motions.",
        replay_initial_position_fields=(
            ("tongs_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "fold_glasses": TaskPreset(
        is_bimanual=False,
        prompt="Fold the glasses and place them into the case.",
        replay_initial_position_fields=(
            ("glass_ori_pose", 7),
            ("box_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "hammer_nail": TaskPreset(
        is_bimanual=False,
        prompt="Use the hammer to drive the nail into the wooden board.",
        replay_initial_position_fields=(
            ("hammer_ori_pose", 7),
            ("nail_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "pick_bucket": TaskPreset(
        is_bimanual=False,
        prompt="Place the boxed food into the bucket and then lift the bucket.",
        replay_initial_position_fields=(
            ("bucket_ori_pose", 7),
            ("boxed_food_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "bimanual_assembly": TaskPreset(
        is_bimanual=True,
        prompt=(
            "Grasp the tray with the left hand and the peg with the right hand, "
            "then insert the peg into the hole."
        ),
        replay_initial_position_fields=(
            ("socket_ori_pose", 7),
            ("peg_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "bimanual_hanoi": TaskPreset(
        is_bimanual=True,
        prompt=(
            "Execute the final two moves of the three-level Tower of Hanoi: "
            "move the medium disk from the middle peg to the right peg with the right hand, "
            "then move the small disk from the left peg to the right peg with the left hand."
        ),
        replay_initial_position_fields=(
            ("hanoi_base_ori_pos", 3),
            ("table_delta_height", 1),
        ),
    ),
    "bimanual_microwave_cook": TaskPreset(
        is_bimanual=True,
        prompt=(
            "Open the microwave door, place the food inside the microwave, "
            "close the door, and press the start button."
        ),
        replay_initial_position_fields=(
            ("hot_dog_ori_pose", 7),
            ("microwave_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "bimanual_photograph": TaskPreset(
        is_bimanual=True,
        prompt=(
            "Grasp the camera with the left hand, align it with the logo, "
            "and press the shutter button with the right hand."
        ),
        replay_initial_position_fields=(
            ("camera_ori_pose", 7),
            ("logo_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
    "bimanual_unlock_ipad": TaskPreset(
        is_bimanual=True,
        prompt="Grasp the iPad and enter the password 123 to unlock the device.",
        replay_initial_position_fields=(
            ("stand_ori_pose", 7),
            ("ipad_ori_pose", 7),
            ("table_delta_height", 1),
        ),
    ),
}
