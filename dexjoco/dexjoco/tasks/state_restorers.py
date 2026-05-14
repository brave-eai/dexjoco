"""Per-task helpers that patch a freshly reset env to match a recorded state[0].

`restore_initial_state` is the public entry point used by the replay script: it
splits the recorded flat proprio vector by `config.proprio_keys`, dispatches to
the task-specific restorer, runs an `mj_forward`, and returns the refreshed
wrapped observation.

Each restorer only writes the fields recorded as proprio (object poses + table
height). Everything else (lighting, table texture, third-person camera) is left
to whatever `env.reset()` produced, so visual randomization stays effective.
"""

import mujoco
import numpy as np


def _split_state_by_proprio(state_vec, proprio_keys, ref_state):
    """Split a flat proprio vector back into a {proprio_key: array} dict.

    Sizes come from a fresh `_compute_observation()` sample because some envs
    declare e.g. `gripper_pose` as `(1,)` in `observation_space` while actually
    returning a 16-d sensor vector; `DexjocoObsAdapter` flattens the real
    array, not the spec.
    """
    parts = {}
    offset = 0
    for key in proprio_keys:
        size = max(1, np.asarray(ref_state[key]).ravel().size)
        parts[key] = np.asarray(state_vec[offset : offset + size], dtype=np.float64)
        offset += size
    return parts


def _apply_delta_h(raw_env, delta_h):
    """Mimic each env's table-height adjustment based on what attributes it caches."""
    raw_env.delta_h = np.float64(delta_h)
    z0 = getattr(raw_env, "_table_body_z0", None)
    if z0 is None:
        z0 = raw_env._table_z
    table_body_id = raw_env._model.body("table").id
    raw_env._model.body_pos[table_body_id, 2] = z0 + delta_h

    # Resolve which leg geoms to adjust:
    #   - most envs cache `_table_leg_geom_ids` (either hardcoded or discovered)
    #   - water_plant caches `_table_leg_half_len0` keyed by leg name instead
    #   - bimanual_microwave_cook caches neither and iterates 4 hardcoded names
    if hasattr(raw_env, "_table_leg_geom_ids"):
        leg_gids = list(raw_env._table_leg_geom_ids)
    else:
        h0_map = getattr(raw_env, "_table_leg_half_len0", None)
        if isinstance(h0_map, dict) and h0_map and isinstance(next(iter(h0_map)), str):
            leg_gids = [raw_env._model.geom(n).id for n in h0_map]
        else:
            leg_gids = [
                raw_env._model.geom(n).id
                for n in ("table_leg_1", "table_leg_2", "table_leg_3", "table_leg_4")
            ]

    if hasattr(raw_env, "_model_geom_pos0"):
        # bimanual_microwave_cook / bimanual_unlock_ipad: shift center down and extend size by half.
        for gid in leg_gids:
            raw_env._model.geom_pos[gid, 2] = raw_env._model_geom_pos0[gid, 2] - 0.5 * delta_h
            raw_env._model.geom_size[gid, 1] = raw_env._model_geom_size0[gid, 1] + 0.5 * delta_h
        return

    # Standard style: extend each leg by the full delta.
    h0_map = raw_env._table_leg_half_len0
    for gid in leg_gids:
        if gid in h0_map:
            h0 = h0_map[gid]
        else:
            name = mujoco.mj_id2name(raw_env._model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            h0 = h0_map[name]
        raw_env._model.geom_size[gid, 1] = h0 + delta_h


def _restore_water_plant(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    spray = parts["spray_ori_pose"]
    raw_env._data.jnt("spray_root").qpos[:3] = spray[:3]
    raw_env._data.jnt("spray_root").qpos[3:7] = spray[3:7]
    raw_env._spray_ori_pose = spray.copy()
    plant = parts["plant_ori_pose"]
    raw_env._model.body("plant").pos = plant[:3]
    raw_env._plant_ori_pose = plant.copy()


def _restore_click_mouse(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    mouse = parts["mouse_ori_pose"]
    raw_env._data.jnt("mouse_root").qpos = mouse
    raw_env.mouse_ori_pose = mouse.copy()
    # env.reset() ties the mousepad's xy to the freshly-sampled mouse_xy;
    # after restoring the mouse to the recorded pose, re-attach the
    # mousepad so the rigid-body contact graph matches.
    from ..sim.envs.panda_click_mouse_env import _MOUSEPAD_OFFSET
    mujoco.mj_forward(raw_env._model, raw_env._data)
    table_z = float(raw_env._data.site_xpos[raw_env._table_site_id][2])
    raw_env._model.body_pos[raw_env._mousepad_body_id][:3] = (
        np.array([mouse[0], mouse[1], table_z]) + _MOUSEPAD_OFFSET
    )
    # The monitor (display_root) has a free joint with `z = z0 + delta_h`;
    # reset() sampled this z with the original delta_h, but `_apply_delta_h`
    # above just rewrote `raw_env.delta_h` to the recorded value, leaving
    # the monitor floating or buried by `(delta_h_original - delta_h_recorded)`.
    # Its xy stays as whatever reset() sampled (not in proprio).
    raw_env._data.jnt("display_root").qpos[2] = (
        raw_env._display_root_z0 + raw_env.delta_h
    )


def _restore_pinch_tongs(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    tongs = parts["tongs_ori_pose"]
    raw_env._data.jnt("tongs_root").qpos = tongs
    raw_env.tongs_ori_pose = tongs.copy()


def _restore_fold_glasses(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    glass = parts["glass_ori_pose"]
    raw_env._data.jnt("glass_root").qpos = glass
    raw_env.glass_ori_pose = glass.copy()
    box = parts["box_ori_pose"]
    box_body_id = raw_env._model.body("open_box").id
    raw_env._model.body_pos[box_body_id] = box[:3]
    raw_env._model.body_quat[box_body_id] = box[3:7]
    raw_env.open_box_ori_pose = box.copy()


def _restore_hammer_nail(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    raw_env._model.body_pos[raw_env._model.body("wood").id, 2] = (
        raw_env._wood_body_z0 + raw_env.delta_h
    )
    hammer = parts["hammer_ori_pose"]
    raw_env._data.jnt("hammer_joint").qpos = hammer
    raw_env.hammer_ori_pose = hammer.copy()
    nail = parts["nail_ori_pose"]
    nail_pos = nail[:3].copy()
    nail_pos[2] = raw_env._nail_body_z0 + raw_env.delta_h
    raw_env._nail_init_pos[:] = nail_pos
    raw_env._data.mocap_pos[raw_env._nail_mocap_id] = nail_pos
    raw_env._data.mocap_quat[raw_env._nail_mocap_id] = nail[3:7]
    raw_env.nail_ori_pose = np.concatenate([nail_pos, nail[3:7]])


def _restore_pick_bucket(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    bucket = parts["bucket_ori_pose"]
    raw_env._data.jnt("bucket_root").qpos = bucket
    raw_env.bucket_ori_pose = bucket.copy()
    raw_env._bucket_z = float(bucket[2])
    boxed_food = parts["boxed_food_ori_pose"]
    raw_env._data.jnt("boxed_food_0_freejoint").qpos = boxed_food
    raw_env.box_food_ori_pose = boxed_food.copy()
    # Re-baseline `_bucket_bottom_z0`: env.reset() captures it at the
    # randomly-sampled bucket pose, but success uses (bottom_z - z0 >= 0.15)
    # so the baseline must reflect the restored bucket pose.
    mujoco.mj_forward(raw_env._model, raw_env._data)
    raw_env._bucket_bottom_z0 = raw_env._data.site_xpos[
        raw_env._bucket_bottom_site_ids, 2
    ].copy()


def _restore_bimanual_assembly(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    socket = parts["socket_ori_pose"]
    raw_env._set_free_joint_pose(
        raw_env._socket_qpos_adr, raw_env._socket_qvel_adr, socket[:3], socket[3:7]
    )
    raw_env._socket_ori_pose = socket.copy()
    peg = parts["peg_ori_pose"]
    raw_env._set_free_joint_pose(
        raw_env._peg_qpos_adr, raw_env._peg_qvel_adr, peg[:3], peg[3:7]
    )
    raw_env._peg_ori_pose = peg.copy()


def _restore_bimanual_hanoi(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    base = np.asarray(parts["hanoi_base_ori_pos"], dtype=np.float64)
    raw_env._model.body_pos[raw_env._base_body_id] = base
    raw_env.base_ori_pos = base.copy()
    # Re-stack disks at the recorded base position; the tower preset matches
    # what reset() would pick, but we need to recompute it relative to the
    # restored base position (otherwise disks float above the prior location).
    base_delta_xy = base[:2] - raw_env._base_init_pos[:2]
    _, tower_state = raw_env._sample_reset_tower_state()
    raw_env._apply_reset_tower_state(tower_state, base_delta_xy)


def _restore_bimanual_microwave_cook(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    hot_dog = parts["hot_dog_ori_pose"]
    raw_env._data.jnt("hot_dog_free").qpos = hot_dog
    raw_env._hot_dog_ori_pose = hot_dog.copy()
    microwave = parts["microwave_ori_pose"]
    raw_env._model.body("microwave_object").pos = microwave[:3]
    raw_env._model.body("microwave_object").quat = microwave[3:7]
    raw_env._microwave_ori_pose = microwave.copy()


def _restore_bimanual_photograph(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    # The target_region geom is reset to (logo_pos + constant offset); preserve
    # that constant offset when we move the logo to the recorded pose.
    target_offset = (
        raw_env._model.geom_pos[raw_env._target_region_geom_id]
        - raw_env._model.geom_pos[raw_env._logo_geom_id]
    ).copy()
    logo = parts["logo_ori_pose"]
    raw_env._model.geom_pos[raw_env._logo_geom_id] = logo[:3]
    raw_env._model.geom_quat[raw_env._logo_geom_id] = logo[3:7]
    raw_env._model.geom_pos[raw_env._target_region_geom_id] = logo[:3] + target_offset
    raw_env.logo_ori_pose = logo.copy()
    camera = parts["camera_ori_pose"]
    raw_env._data.jnt("camera_root").qpos = camera
    raw_env.camera_ori_pose = camera.copy()


def _restore_bimanual_unlock_ipad(raw_env, parts):
    _apply_delta_h(raw_env, float(parts["table_delta_height"][0]))
    stand = parts["stand_ori_pose"]
    stand_body_id = raw_env._model.body("ipad_stand").id
    raw_env._model.body_pos[stand_body_id] = stand[:3]
    raw_env._stand_ori_pose = stand.copy()
    ipad = parts["ipad_ori_pose"]
    raw_env._data.jnt("ipad_freejoint").qpos[:3] = ipad[:3]
    raw_env._ipad_ori_pose = ipad.copy()


_STATE_RESTORERS = {
    "water_plant": _restore_water_plant,
    "click_mouse": _restore_click_mouse,
    "pinch_tongs": _restore_pinch_tongs,
    "fold_glasses": _restore_fold_glasses,
    "hammer_nail": _restore_hammer_nail,
    "pick_bucket": _restore_pick_bucket,
    "bimanual_assembly": _restore_bimanual_assembly,
    "bimanual_hanoi": _restore_bimanual_hanoi,
    "bimanual_microwave_cook": _restore_bimanual_microwave_cook,
    "bimanual_photograph": _restore_bimanual_photograph,
    "bimanual_unlock_ipad": _restore_bimanual_unlock_ipad,
}


def has_restorer(task_id):
    """True if `task_id` has a registered initial-state restorer."""
    return task_id in _STATE_RESTORERS


def restore_initial_state(env, task_id, config, state_vec):
    """Patch a freshly reset env to match the recorded `state[0]`.

    Args:
        env: a wrapped env (e.g. the one returned by `TaskConfig.get_environment`).
        task_id: key into the task registry; must satisfy `has_restorer(task_id)`.
        config: the matching `TaskConfig` (used for `proprio_keys`).
        state_vec: 1-D recorded proprio vector for `t = 0`.

    Returns:
        The refreshed wrapped observation produced after the restore + mj_forward.
    """
    raw_env = env.unwrapped
    ref_state = raw_env._compute_observation()["state"]
    parts = _split_state_by_proprio(state_vec, config.proprio_keys, ref_state)
    _STATE_RESTORERS[task_id](raw_env, parts)
    mujoco.mj_forward(raw_env._model, raw_env._data)
    return env.observation(raw_env._compute_observation())
