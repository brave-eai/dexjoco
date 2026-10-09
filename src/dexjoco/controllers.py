import mujoco
import numpy as np
from dm_robotics.transformations import transformations as tr


def pd_control(x, x_des, dx, kp_kv, ddx_max=0.0):
    x_err = x - x_des
    dx_err = dx
    x_err *= -kp_kv[:, 0]
    dx_err *= -kp_kv[:, 1]
    if ddx_max > 0.0:
        x_err_sq_norm = np.sum(x_err**2)
        ddx_max_sq = ddx_max**2
        if x_err_sq_norm > ddx_max_sq:
            x_err *= ddx_max / np.sqrt(x_err_sq_norm)
    return x_err + dx_err


def pd_control_orientation(quat, quat_des, w, kp_kv, dw_max=0.0):
    quat_err = tr.quat_diff_active(source_quat=quat_des, target_quat=quat)
    ori_err = tr.quat_to_axisangle(quat_err)
    w_err = w
    ori_err *= -kp_kv[:, 0]
    w_err *= -kp_kv[:, 1]
    if dw_max > 0.0:
        ori_err_sq_norm = np.sum(ori_err**2)
        dw_max_sq = dw_max**2
        if ori_err_sq_norm > dw_max_sq:
            ori_err *= dw_max / np.sqrt(ori_err_sq_norm)
    return ori_err + w_err


def opspace(
    model,
    data,
    site_id,
    dof_ids,
    pos,
    ori=None,
    joint=None,
    pos_gains=(200.0, 200.0, 200.0),
    ori_gains=(200.0, 200.0, 200.0),
    damping_ratio=1.0,
    nullspace_stiffness=0.5,
    max_pos_acceleration=None,
    max_ori_acceleration=None,
    gravity_comp=True,
):
    x_des = np.asarray(pos)
    if ori is None:
        quat_des = np.asarray(tr.mat_to_quat(data.site_xmat[site_id].reshape(3, 3)))
    else:
        ori = np.asarray(ori)
        quat_des = np.asarray(tr.mat_to_quat(ori)) if ori.shape == (3, 3) else ori
    q_des = data.qpos[dof_ids] if joint is None else np.asarray(joint)

    kp = np.asarray(pos_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_pos = np.stack([kp, kd], axis=-1)
    kp = np.asarray(ori_gains)
    kd = damping_ratio * 2 * np.sqrt(kp)
    kp_kv_ori = np.stack([kp, kd], axis=-1)
    kp_joint = np.full((len(dof_ids),), nullspace_stiffness)
    kd_joint = damping_ratio * 2 * np.sqrt(kp_joint)
    kp_kv_joint = np.stack([kp_joint, kd_joint], axis=-1)

    ddx_max = max_pos_acceleration if max_pos_acceleration is not None else 0.0
    dw_max = max_ori_acceleration if max_ori_acceleration is not None else 0.0
    q = data.qpos[dof_ids]
    dq = data.qvel[dof_ids]
    J_v = np.zeros((3, model.nv), dtype=np.float64)
    J_w = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, J_v, J_w, site_id)  # pyright: ignore[reportAttributeAccessIssue]
    J_v = J_v[:, dof_ids]
    J_w = J_w[:, dof_ids]
    J = np.concatenate([J_v, J_w], axis=0)

    ddx = pd_control(
        x=data.site_xpos[site_id], x_des=x_des, dx=J_v @ dq,
        kp_kv=kp_kv_pos, ddx_max=ddx_max,
    )
    quat = np.asarray(tr.mat_to_quat(data.site_xmat[site_id].reshape(3, 3)))
    if quat @ quat_des < 0.0:
        quat *= -1.0
    dw = pd_control_orientation(
        quat=quat, quat_des=quat_des, w=J_w @ dq,
        kp_kv=kp_kv_ori, dw_max=dw_max,
    )

    M = np.zeros((model.nv, model.nv), dtype=np.float64)
    mujoco.mj_fullM(model, M, data.qM)  # pyright: ignore[reportAttributeAccessIssue]
    M = M[dof_ids, :][:, dof_ids]
    M_inv = np.linalg.inv(M)
    Mx_inv = J @ M_inv @ J.T
    if abs(np.linalg.det(Mx_inv)) >= 1e-2:
        Mx = np.linalg.inv(Mx_inv)
    else:
        Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

    ddx_dw = np.concatenate([ddx, dw], axis=0)
    tau = J.T @ Mx @ ddx_dw
    ddq = pd_control(
        x=q, x_des=q_des, dx=dq,
        kp_kv=kp_kv_joint, ddx_max=0.0,
    )
    Jnull = M_inv @ J.T @ Mx
    tau += (np.eye(len(q)) - J.T @ Jnull.T) @ ddq
    if gravity_comp:
        tau += data.qfrc_bias[dof_ids]
    return tau
