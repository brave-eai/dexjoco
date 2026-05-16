"""Per-step depth capture for Dexjoco envs, and helpers to write depth
videos alongside the RGB captures.

The depth image is what MuJoCo calls the "linearised z-buffer": the
distance from the camera plane to the closest surface, in metres. It is
not Euclidean distance to the optical centre.

Each env's `_compute_observation()` already calls `render()` to produce
RGB frames; this module mirrors that call but with `render_mode=
"depth_array"` so the camera-id ordering matches the dict ordering of
`obs["images"]` exactly.

Writing:
- `<name>_depth.npz`: a compressed archive with a single float32 array
  named "depth" of shape (T, H, W), in metres. Use this for training or
  any consumer that needs the true depth.
- `<name>_depth.mp4`: an 8-bit gray-scale H.264 preview, normalised to
  the per-clip min/max of valid depth pixels. Lossy.

Invalid pixels (depth equal to mujoco's far plane, returned when no
geom is hit) are kept as float `inf` in the npz so consumers can mask
them, and clipped out of the normalisation when building the mp4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexjoco.data.video_writer import Mp4VideoWriter


def _env_viewer_and_camera_ids(raw_env, image_keys):
    """Return (viewer, [camera_id_per_image_key]) for the given env.

    All envs except water_plant store the renderer at `_viewer` and the
    ordered camera-id tuple at `camera_id`. water_plant uses `_mj_viewer`
    and exposes the two ids individually. We probe in that order.
    """
    viewer = getattr(raw_env, "_viewer", None)
    if viewer is None:
        viewer = getattr(raw_env, "_mj_viewer", None)
    if viewer is None:
        return None, None

    cam_ids = getattr(raw_env, "camera_id", None)
    if cam_ids is not None:
        cam_ids = tuple(cam_ids)
    else:
        wrist = getattr(raw_env, "_wrist_camera_id", None)
        front = getattr(raw_env, "_front_camera_id", None)
        if wrist is None or front is None or len(image_keys) != 2:
            return viewer, None
        # water_plant render() returns [wrist, front/random_camera].
        cam_ids = (wrist, front)

    if len(cam_ids) != len(image_keys):
        return viewer, None
    return viewer, cam_ids


def _linearise_depth(zbuf, znear, zfar):
    """Convert mujoco's nonlinear depth buffer (`mjr_readPixels` returns the
    OpenGL z-buffer in [0, 1]) into linear depth in metres along the camera
    ray axis. Pixels at the far plane (z == 1) become `inf`."""
    z = np.asarray(zbuf, dtype=np.float32)
    denom = zfar - z * (zfar - znear)
    out = np.full_like(z, np.inf, dtype=np.float32)
    mask = denom > 0
    out[mask] = (znear * zfar / denom)[mask]
    return out


def collect_depth_frames(env, image_keys):
    """Render a depth_array for each RGB camera key in `image_keys`.

    Returns a dict mapping the same key (e.g. "wrist") to a float32
    HxW depth image in metres, or an empty dict if depth capture is
    not available for this env.
    """
    raw = env.unwrapped
    viewer, cam_ids = _env_viewer_and_camera_ids(raw, image_keys)
    if viewer is None or cam_ids is None:
        return {}
    extent = float(raw._model.stat.extent)
    znear = float(raw._model.vis.map.znear) * extent
    zfar = float(raw._model.vis.map.zfar) * extent
    out = {}
    for key, cid in zip(image_keys, cam_ids):
        zbuf = viewer.render(render_mode="depth_array", camera_id=cid)
        out[key] = _linearise_depth(zbuf, znear, zfar)
    return out


def _normalise_depth_for_preview(depth_frames):
    """Stack and 8-bit-quantise a list of float depth frames for an mp4
    preview. The per-clip min/max are taken over finite pixels so very
    distant 'sky' pixels don't crush all the detail to a single bin."""
    stack = np.stack(depth_frames, axis=0)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(finite.min())
        hi = float(finite.max())
        if hi <= lo:
            hi = lo + 1e-6
    clipped = np.clip(stack, lo, hi)
    norm = (clipped - lo) / (hi - lo)
    norm[~np.isfinite(stack)] = 0.0
    gray = (norm * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1), lo, hi


def write_depth_outputs(depth_frames_per_camera, videos_dir, video_fps):
    """For each camera, write `<cam>_depth.npz` (float32 metres) and
    `<cam>_depth.mp4` (gray H.264 preview, normalised per clip).

    `depth_frames_per_camera` is a dict mapping camera key to a list of
    per-step depth frames. The lists may be empty (camera was never
    rendered) - in which case nothing is written for that key.
    """
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for cam_key, frames in depth_frames_per_camera.items():
        if not frames:
            continue
        depth_stack = np.stack(frames, axis=0).astype(np.float32)
        npz_path = videos_dir / f"{cam_key}_depth.npz"
        np.savez_compressed(npz_path, depth=depth_stack)
        saved.append(str(npz_path))

        preview, lo, hi = _normalise_depth_for_preview(frames)
        mp4_path = videos_dir / f"{cam_key}_depth.mp4"
        writer = Mp4VideoWriter.create_h264(
            fps=video_fps,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=21,
            thread_type="FRAME",
            thread_count=2,
        )
        writer.start(str(mp4_path))
        for frame in preview:
            writer.write_frame(frame)
        writer.stop()
        saved.append(f"{mp4_path} (depth range [{lo:.3f}, {hi:.3f}] m)")
    return saved
