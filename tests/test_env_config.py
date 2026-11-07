"""Check task features and the EnvHub configuration contract."""

import json

import pytest

from dexjoco.configs import DexJoCoEnvConfig
from dexjoco.env_factory import TASK_IDS


@pytest.mark.parametrize("task", list(TASK_IDS))
@pytest.mark.parametrize("action_format", ["quaternion", "rotvec"])
def test_infer_arm_configuration(task, action_format):
    cfg = DexJoCoEnvConfig(task=task, action_format=action_format)
    bimanual = task.startswith("bimanual_")
    assert cfg.bimanual is bimanual
    assert cfg.features["agent_pos"].shape == (46 if bimanual else 23,)
    arm_dim = 23 if action_format == "quaternion" else 22
    assert cfg.features["action"].shape == (arm_dim * (2 if bimanual else 1),)
    assert ("pixels/wrist_left" in cfg.features) is bimanual
    assert ("pixels/wrist_right" in cfg.features) is bimanual
    assert ("pixels/wrist" in cfg.features) is not bimanual


@pytest.mark.parametrize("task,bimanual", [("water_plant", True), ("bimanual_hanoi", False)])
def test_reject_conflicting_arm_configuration(task, bimanual):
    with pytest.raises(ValueError, match="conflicts with task"):
        DexJoCoEnvConfig(task=task, bimanual=bimanual)


def test_envhub_options_round_trip() -> None:
    cfg = DexJoCoEnvConfig(
        task="bimanual_hanoi",
        action_format="rotvec",
        randomize=True,
        randomize_dynamics=True,
        offsamples=0,
        render_width=320,
        render_height=240,
    )
    options = json.loads(json.dumps(cfg.to_make_env_kwargs()))
    assert DexJoCoEnvConfig(**options) == cfg
