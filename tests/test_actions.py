"""Exercise action representations, per-arm ordering, and identity rotations."""

import numpy as np
import pytest

from dexjoco.actions import quaternion_to_rotvec, rotvec_to_quaternion


@pytest.mark.parametrize("bimanual", [False, True])
def test_action_representations(bimanual: bool):
    arms = 2 if bimanual else 1
    actions = np.zeros((2, 23 * arms), dtype=np.float64)
    for arm in range(arms):
        offset = arm * 23
        actions[0, offset + 3] = 1.0
        actions[1, offset : offset + 3] = [0.2, -0.3, 0.4]
        actions[1, offset + 3] = np.sqrt(0.5)
        actions[1, offset + 4 + arm] = np.sqrt(0.5)
        actions[:, offset + 7 : offset + 23] = np.arange(16) / 20 + arm
    converted = quaternion_to_rotvec(actions, bimanual=bimanual)
    assert converted.dtype == np.float64
    assert converted.shape == (2, 22 * arms)
    for arm in range(arms):
        np.testing.assert_array_equal(converted[0, 22 * arm : 22 * arm + 6], np.zeros(6))
        expected = np.zeros(3)
        expected[arm] = np.pi / 2
        np.testing.assert_allclose(converted[1, 22 * arm + 3 : 22 * arm + 6], expected, atol=1e-15)
        np.testing.assert_array_equal(
            converted[:, 22 * arm + 6 : 22 * arm + 22], actions[:, 23 * arm + 7 : 23 * arm + 23]
        )
    restored = rotvec_to_quaternion(converted, bimanual=bimanual)
    np.testing.assert_allclose(restored, actions, atol=1e-15, rtol=0)


@pytest.mark.parametrize("bimanual", [False, True])
def test_zero_rotvec_is_identity(bimanual: bool):
    arms = 2 if bimanual else 1
    converted = rotvec_to_quaternion(np.zeros((1, 22 * arms)), bimanual=bimanual)
    for arm in range(arms):
        np.testing.assert_array_equal(converted[0, arm * 23 + 3 : arm * 23 + 7], [1, 0, 0, 0])
    with pytest.raises(ValueError, match="zero norm"):
        quaternion_to_rotvec(np.zeros((1, 23 * arms)), bimanual=bimanual)
