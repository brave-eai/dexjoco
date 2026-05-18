from pathlib import Path

import flax.nnx as nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
import tyro

import openpi.models.model as _model


def main(
    input_path: Path,
    output_path: Path,
) -> None:

    input_params_path = Path(input_path) / "params"
    output_params_path = Path(output_path) / "params"

    params = _model.restore_params(str(input_params_path), restore_type=np.ndarray)

    action_in_kernel = params["action_in_proj"]["kernel"]
    action_out_kernel = params["action_out_proj"]["kernel"]
    action_out_bias = params["action_out_proj"]["bias"]

    if action_in_kernel.ndim != 2:
        raise ValueError(f"Expected action_in_proj.kernel ndim=2, got {action_in_kernel.ndim}")
    if action_out_kernel.ndim != 2:
        raise ValueError(f"Expected action_out_proj.kernel ndim=2, got {action_out_kernel.ndim}")
    if action_out_bias.ndim != 1:
        raise ValueError(f"Expected action_out_proj.bias ndim=1, got {action_out_bias.ndim}")
    if action_in_kernel.shape[0] != 32 or action_out_kernel.shape[1] != 32 or action_out_bias.shape[0] != 32:
        raise ValueError(
            "Expected action_dim=32 checkpoint, got "
            f"action_in_proj.kernel.shape={action_in_kernel.shape}, "
            f"action_out_proj.kernel.shape={action_out_kernel.shape}, "
            f"action_out_proj.bias.shape={action_out_bias.shape}"
        )

    target_action_dim = 44

    hidden_dim = action_in_kernel.shape[1]

    # Keep first 32 dims from pretrained weights, and randomly initialize new dims (32:44)
    # with the same default NNX Linear initialization used by Pi0.
    in_proj_default = nnx.Linear(target_action_dim, hidden_dim, rngs=nnx.Rngs(jax.random.key(0)))
    out_proj_default = nnx.Linear(hidden_dim, target_action_dim, rngs=nnx.Rngs(jax.random.key(1)))

    init_action_in_kernel = np.asarray(in_proj_default.kernel.value, dtype=action_in_kernel.dtype).copy()
    init_action_in_kernel[:32, :] = action_in_kernel
    params["action_in_proj"]["kernel"] = init_action_in_kernel

    init_action_out_kernel = np.asarray(out_proj_default.kernel.value, dtype=action_out_kernel.dtype).copy()
    init_action_out_kernel[:, :32] = action_out_kernel
    params["action_out_proj"]["kernel"] = init_action_out_kernel

    init_action_out_bias = np.asarray(out_proj_default.bias.value, dtype=action_out_bias.dtype).copy()
    init_action_out_bias[:32] = action_out_bias
    params["action_out_proj"]["bias"] = init_action_out_bias

    output_params_path.parent.mkdir(parents=True, exist_ok=False)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(
            str(output_params_path),
            args=ocp.args.PyTreeSave(item={"params": params}),  # type: ignore
        )

    print(f"Saved converted params to: {output_params_path}")
    print(
        "action_in_proj.kernel: "
        f"{action_in_kernel.shape} -> {params['action_in_proj']['kernel'].shape}, "
        "action_out_proj.kernel: "
        f"{action_out_kernel.shape} -> {params['action_out_proj']['kernel'].shape}, "
        "action_out_proj.bias: "
        f"{action_out_bias.shape} -> {params['action_out_proj']['bias'].shape}"
    )


if __name__ == "__main__":
    tyro.cli(main)


"""
Example:
python scripts/convert_to_action_dim_44_model.py \
  --input-path ../checkpoints/pi05_base \
  --output-path ../checkpoints/pi05_base_action_dim_44
"""
