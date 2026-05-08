from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import glfw
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

INTERVENTION_START_DELAY_SECONDS = 2.0


@dataclass(frozen=True)
class SingleArmTeleopConfig:
    vive_udp_host: str = "127.0.0.1"
    vive_udp_port: int = 5012
    hand_udp_host: str = "127.0.0.1"
    hand_udp_port: int = 5014
    pose_scale: float = 1.5


@dataclass(frozen=True)
class BimanualTeleopConfig:
    vive_udp_host: str = "127.0.0.1"
    vive_udp_port: int = 5012
    right_hand_udp_host: str = "127.0.0.1"
    right_hand_udp_port: int = 5014
    left_hand_udp_host: str = "127.0.0.1"
    left_hand_udp_port: int = 5016
    pose_scale: float = 1.5


def _quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(matrix).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _create_udp_socket(ip, port, timeout=0.2, reuseaddr=False):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if reuseaddr:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((ip, port))
    sock.settimeout(timeout)
    return sock


def _get_mujoco_window(env):
    for viewer_attr in ("_mj_viewer", "_viewer"):
        viewer = getattr(env, viewer_attr, None)
        if viewer is None:
            continue
        window = getattr(viewer, "window", None)
        if window is not None:
            return window
        nested_window = getattr(getattr(viewer, "viewer", None), "window", None)
        if nested_window is not None:
            return nested_window
    return None


def _maybe_set_key_callback(env, callback, current_window=None):
    window = _get_mujoco_window(env)
    if window is None:
        return current_window, False
    if window is current_window:
        return current_window, True
    glfw.set_key_callback(window, callback)
    return window, True


class SingleArmViveHandTeleopWrapper(gym.ActionWrapper):
    def __init__(self, env, config: SingleArmTeleopConfig):
        super().__init__(env)
        self.config = config
        self.intervened = False
        self.reset_trigger = False
        self.tracker_start_world = None
        self.ee_start = None
        self.latest_tracker = np.eye(4, dtype=np.float64)
        self.latest_allegro_angles = np.zeros(16, dtype=np.float64)

        self._tracker_lock = threading.Lock()
        self._hand_lock = threading.Lock()
        self._stop_tracker_thread = False
        self._stop_hand_thread = False
        self._key_callback_window = None

        self._vive_socket = _create_udp_socket(
            config.vive_udp_host,
            config.vive_udp_port,
            timeout=0.5,
        )
        self._hand_socket = _create_udp_socket(
            config.hand_udp_host,
            config.hand_udp_port,
            timeout=0.1,
            reuseaddr=True,
        )

        self._vive_thread = threading.Thread(target=self._recv_vive_loop, daemon=True)
        self._hand_thread = threading.Thread(target=self._recv_hand_loop, daemon=True)
        self._vive_thread.start()
        self._hand_thread.start()

        self._ensure_key_callback()

    def _ensure_key_callback(self):
        self._key_callback_window, _ = _maybe_set_key_callback(
            self.env,
            self.glfw_on_key,
            current_window=self._key_callback_window,
        )

    def _recv_vive_loop(self):
        while not self._stop_tracker_thread:
            try:
                data, _ = self._vive_socket.recvfrom(2048)
                if len(data) < 12 * 8:
                    continue
                pose = np.frombuffer(data, dtype=np.float64, count=12).reshape(3, 4)
                transform = np.eye(4, dtype=np.float64)
                transform[:3, :] = pose
                with self._tracker_lock:
                    self.latest_tracker = transform
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

    def _recv_hand_loop(self):
        while not self._stop_hand_thread:
            try:
                data, _ = self._hand_socket.recvfrom(4096)
                if not data:
                    continue
                hand_angles = np.frombuffer(data, dtype=np.float64)
                parsed = np.zeros(16, dtype=np.float64)
                parsed[: min(parsed.size, hand_angles.size)] = hand_angles[: parsed.size]
                with self._hand_lock:
                    self.latest_allegro_angles = parsed
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

    def glfw_on_key(self, window, key, scancode, action, mods):
        del window
        del scancode
        del mods

        if action == glfw.PRESS and key == glfw.KEY_SEMICOLON:
            self.intervened = not self.intervened
            if self.intervened:
                print(
                    f"Teleoperation will be enabled in "
                    f"{INTERVENTION_START_DELAY_SECONDS:.1f}s..."
                )
                time.sleep(INTERVENTION_START_DELAY_SECONDS)
                with self._tracker_lock:
                    self.tracker_start_world = self.latest_tracker.copy()
                try:
                    self.ee_start = self.env.get_end_effector_pose_matrix().copy()
                except Exception:
                    self.ee_start = None
                    self.intervened = False
                    print("Warning: failed to read EE pose; teleop was not enabled.")
            else:
                self.tracker_start_world = None
                self.ee_start = None
            print(f"Teleoperation {'enabled' if self.intervened else 'disabled'}.")

        if action == glfw.PRESS and key == glfw.KEY_R:
            self.reset_trigger = True

    def _vive_action(self):
        if not self.intervened or self.tracker_start_world is None or self.ee_start is None:
            return None

        with self._tracker_lock:
            tracker_now = self.latest_tracker.copy()

        try:
            tracker_delta = np.linalg.inv(self.tracker_start_world) @ tracker_now
        except np.linalg.LinAlgError:
            return None

        tracker_delta[:3, 3] *= self.config.pose_scale
        target_pose = self.ee_start @ tracker_delta
        return np.concatenate(
            [target_pose[:3, 3], _quat_wxyz_from_matrix(target_pose[:3, :3])],
            axis=0,
        )

    def _hold_pose_action(self):
        ee_pose = self.env.get_end_effector_pose_matrix()
        return np.concatenate(
            [ee_pose[:3, 3], _quat_wxyz_from_matrix(ee_pose[:3, :3])],
            axis=0,
        )

    def _hand_action(self, action: np.ndarray):
        if self.intervened:
            with self._hand_lock:
                return self.latest_allegro_angles.copy()
        if action is not None and action.shape[0] >= 23:
            return np.asarray(action[7:23], dtype=np.float64)
        return np.zeros(16, dtype=np.float64)

    def action(self, action: np.ndarray):
        pose_action = self._vive_action()
        if pose_action is None:
            pose_action = self._hold_pose_action()
        hand_action = self._hand_action(action)
        return np.concatenate([pose_action, hand_action], axis=0)

    def step(self, action):
        self._ensure_key_callback()
        teleop_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(teleop_action)
        info = dict(info)
        info["intervene_action"] = teleop_action
        if self.reset_trigger:
            info["manual_reset"] = True
            self.reset_trigger = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ensure_key_callback()
        self.intervened = False
        self.reset_trigger = False
        self.tracker_start_world = None
        self.ee_start = None
        with self._hand_lock:
            self.latest_allegro_angles = np.zeros(16, dtype=np.float64)
        return obs, info

    def close(self):
        self._stop_tracker_thread = True
        self._stop_hand_thread = True
        if self._vive_thread.is_alive():
            self._vive_thread.join(timeout=1.0)
        if self._hand_thread.is_alive():
            self._hand_thread.join(timeout=1.0)
        try:
            self._vive_socket.close()
        except Exception:
            pass
        try:
            self._hand_socket.close()
        except Exception:
            pass
        self.env.close()


class DualArmViveHandTeleopWrapper(gym.ActionWrapper):
    def __init__(self, env, config: BimanualTeleopConfig):
        super().__init__(env)
        self.config = config
        self.intervened = False
        self.reset_trigger = False
        self.tracker_start_world = None
        self.ee_start = None
        self.latest_tracker_right = np.eye(4, dtype=np.float64)
        self.latest_tracker_left = np.eye(4, dtype=np.float64)
        self.latest_allegro_angles_right = np.zeros(16, dtype=np.float64)
        self.latest_allegro_angles_left = np.zeros(16, dtype=np.float64)

        self._tracker_lock = threading.Lock()
        self._hand_lock = threading.Lock()
        self._stop_tracker_thread = False
        self._stop_hand_thread = False
        self._key_callback_window = None

        self._vive_socket = _create_udp_socket(
            config.vive_udp_host,
            config.vive_udp_port,
            timeout=0.5,
        )
        self._right_hand_socket = _create_udp_socket(
            config.right_hand_udp_host,
            config.right_hand_udp_port,
            timeout=0.1,
            reuseaddr=True,
        )
        self._left_hand_socket = _create_udp_socket(
            config.left_hand_udp_host,
            config.left_hand_udp_port,
            timeout=0.1,
            reuseaddr=True,
        )

        self._vive_thread = threading.Thread(target=self._recv_vive_loop, daemon=True)
        self._hand_thread = threading.Thread(target=self._recv_hand_loop, daemon=True)
        self._vive_thread.start()
        self._hand_thread.start()

        self._ensure_key_callback()

    def _ensure_key_callback(self):
        self._key_callback_window, _ = _maybe_set_key_callback(
            self.env,
            self.glfw_on_key,
            current_window=self._key_callback_window,
        )

    def _recv_vive_loop(self):
        while not self._stop_tracker_thread:
            try:
                data, _ = self._vive_socket.recvfrom(2048)
                if len(data) < 12 * 8:
                    continue
                if len(data) >= 24 * 8:
                    arr = np.frombuffer(data, dtype=np.float64, count=24)
                    right_pose = arr[:12].reshape(3, 4)
                    left_pose = arr[12:24].reshape(3, 4)
                else:
                    pose = np.frombuffer(data, dtype=np.float64, count=12).reshape(3, 4)
                    right_pose = pose
                    left_pose = pose

                right_transform = np.eye(4, dtype=np.float64)
                right_transform[:3, :] = right_pose
                left_transform = np.eye(4, dtype=np.float64)
                left_transform[:3, :] = left_pose
                with self._tracker_lock:
                    self.latest_tracker_right = right_transform
                    self.latest_tracker_left = left_transform
            except socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

    def _recv_hand_loop(self):
        while not self._stop_hand_thread:
            try:
                right_data, _ = self._right_hand_socket.recvfrom(4096)
                if right_data:
                    right_angles = np.frombuffer(right_data, dtype=np.float64)
                    parsed_right = np.zeros(16, dtype=np.float64)
                    parsed_right[: min(parsed_right.size, right_angles.size)] = right_angles[
                        : parsed_right.size
                    ]
                    with self._hand_lock:
                        self.latest_allegro_angles_right = parsed_right
            except socket.timeout:
                pass
            except Exception:
                time.sleep(0.01)

            try:
                left_data, _ = self._left_hand_socket.recvfrom(4096)
                if left_data:
                    left_angles = np.frombuffer(left_data, dtype=np.float64)
                    parsed_left = np.zeros(16, dtype=np.float64)
                    parsed_left[: min(parsed_left.size, left_angles.size)] = left_angles[
                        : parsed_left.size
                    ]
                    with self._hand_lock:
                        self.latest_allegro_angles_left = parsed_left
            except socket.timeout:
                pass
            except Exception:
                time.sleep(0.01)

    def glfw_on_key(self, window, key, scancode, action, mods):
        del window
        del scancode
        del mods

        if action == glfw.PRESS and key == glfw.KEY_SEMICOLON:
            self.intervened = not self.intervened
            if self.intervened:
                print(
                    f"Teleoperation will be enabled in "
                    f"{INTERVENTION_START_DELAY_SECONDS:.1f}s..."
                )
                time.sleep(INTERVENTION_START_DELAY_SECONDS)
                with self._tracker_lock:
                    self.tracker_start_world = {
                        "right": self.latest_tracker_right.copy(),
                        "left": self.latest_tracker_left.copy(),
                    }
                try:
                    right_pose, left_pose = self.env.get_end_effector_pose_matrix()
                    self.ee_start = {
                        "right": right_pose.copy(),
                        "left": left_pose.copy(),
                    }
                except Exception:
                    self.ee_start = None
                    self.intervened = False
                    print("Warning: failed to read EE pose; teleop was not enabled.")
            else:
                self.tracker_start_world = None
                self.ee_start = None
            print(f"Teleoperation {'enabled' if self.intervened else 'disabled'}.")

        if action == glfw.PRESS and key == glfw.KEY_R:
            self.reset_trigger = True

    def _vive_action(self):
        if not self.intervened or self.tracker_start_world is None or self.ee_start is None:
            return None

        with self._tracker_lock:
            tracker_right = self.latest_tracker_right.copy()
            tracker_left = self.latest_tracker_left.copy()

        try:
            tracker_delta_right = np.linalg.inv(self.tracker_start_world["right"]) @ tracker_right
            tracker_delta_left = np.linalg.inv(self.tracker_start_world["left"]) @ tracker_left
        except np.linalg.LinAlgError:
            return None

        tracker_delta_right[:3, 3] *= self.config.pose_scale
        tracker_delta_left[:3, 3] *= self.config.pose_scale
        target_right = self.ee_start["right"] @ tracker_delta_right
        target_left = self.ee_start["left"] @ tracker_delta_left

        return {
            "right": np.concatenate(
                [target_right[:3, 3], _quat_wxyz_from_matrix(target_right[:3, :3])],
                axis=0,
            ),
            "left": np.concatenate(
                [target_left[:3, 3], _quat_wxyz_from_matrix(target_left[:3, :3])],
                axis=0,
            ),
        }

    def _hold_pose_action(self):
        right_pose, left_pose = self.env.get_end_effector_pose_matrix()
        return {
            "right": np.concatenate(
                [right_pose[:3, 3], _quat_wxyz_from_matrix(right_pose[:3, :3])],
                axis=0,
            ),
            "left": np.concatenate(
                [left_pose[:3, 3], _quat_wxyz_from_matrix(left_pose[:3, :3])],
                axis=0,
            ),
        }

    def _hand_action(self, action: np.ndarray):
        if self.intervened:
            with self._hand_lock:
                return (
                    self.latest_allegro_angles_right.copy(),
                    self.latest_allegro_angles_left.copy(),
                )

        hand_action = action[14:] if (action is not None and action.shape[0] > 14) else np.array([], dtype=np.float64)
        if hand_action.size >= 32:
            return (
                np.asarray(hand_action[:16], dtype=np.float64),
                np.asarray(hand_action[16:32], dtype=np.float64),
            )
        if hand_action.size >= 16:
            return (
                np.asarray(hand_action[:16], dtype=np.float64),
                np.zeros(16, dtype=np.float64),
            )
        return np.zeros(16, dtype=np.float64), np.zeros(16, dtype=np.float64)

    def action(self, action: np.ndarray):
        pose_action = self._vive_action()
        if pose_action is None:
            pose_action = self._hold_pose_action()

        hand_right, hand_left = self._hand_action(action)
        return {
            "right": np.concatenate([pose_action["right"], hand_right], axis=0),
            "left": np.concatenate([pose_action["left"], hand_left], axis=0),
        }

    def step(self, action):
        self._ensure_key_callback()
        teleop_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(teleop_action)
        info = dict(info)
        info["intervene_action"] = teleop_action
        if self.reset_trigger:
            info["manual_reset"] = True
            self.reset_trigger = False
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ensure_key_callback()
        self.intervened = False
        self.reset_trigger = False
        self.tracker_start_world = None
        self.ee_start = None
        with self._hand_lock:
            self.latest_allegro_angles_right = np.zeros(16, dtype=np.float64)
            self.latest_allegro_angles_left = np.zeros(16, dtype=np.float64)
        return obs, info

    def close(self):
        self._stop_tracker_thread = True
        self._stop_hand_thread = True
        if self._vive_thread.is_alive():
            self._vive_thread.join(timeout=1.0)
        if self._hand_thread.is_alive():
            self._hand_thread.join(timeout=1.0)
        for sock in (
            self._vive_socket,
            self._right_hand_socket,
            self._left_hand_socket,
        ):
            try:
                sock.close()
            except Exception:
                pass
        self.env.close()
