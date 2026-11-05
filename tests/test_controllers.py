"""Check the orientation controller's world-frame rotation error."""

import numpy as np
from scipy.spatial.transform import Rotation

from dexjoco.controllers import pd_control_orientation


def test_orientation_error_order_and_damping():
    current = Rotation.from_euler("y", 90, degrees=True).as_quat(scalar_first=True)
    desired = Rotation.from_euler("x", 90, degrees=True).as_quat(scalar_first=True)
    gains = np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
    velocity = np.array([0.1, 0.2, 0.3])
    error = np.array([-1.0, 1.0, 1.0]) * (2 * np.pi / (3 * np.sqrt(3)))
    result = pd_control_orientation(current, desired, velocity.copy(), gains)
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float64), -2 * error - 3 * velocity, atol=1e-14
    )


def test_quaternion_sign_and_scale():
    current = Rotation.from_rotvec([0.2, -0.3, 0.4]).as_quat(scalar_first=True)
    desired = Rotation.from_rotvec([-0.1, 0.2, 0.3]).as_quat(scalar_first=True)
    gains = np.ones((3, 2))
    expected = pd_control_orientation(current, desired, np.zeros(3), gains)
    actual = pd_control_orientation(-2 * current, 3 * desired, np.zeros(3), gains)
    np.testing.assert_allclose(actual, expected, atol=1e-15)
