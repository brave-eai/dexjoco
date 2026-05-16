from dataclasses import dataclass
import subprocess
from typing import Literal

import tyro

"""
Examples:

python scripts/launch_tmux_train.py \
  --config-names hammer_nail click_mouse \
  --gpus 0 1 \
  --wandb-project dexjoco-openpi \
  --wandb-mode offline

python scripts/launch_tmux_train.py \
  --config-names hammer_nail_rand_full click_mouse_rand_full \
  --gpus 0 1 \
  --wandb-project dexjoco-openpi-rand-full \
  --wandb-mode offline

4 tasks each using 2 GPUs:
python scripts/launch_tmux_train.py \
  --config-names bimanual_assembly bimanual_unlock_ipad bimanual_microwave_cook bimanual_hanoi \
  --gpus 0,1 2,3 4,5 6,7 \
  --wandb-project dexjoco-openpi \
  --wandb-mode offline \
  --nccl-p2p-disable
"""


@dataclass
class Args:
    config_names: list[str]
    gpus: list[str]
    wandb_project: str = "dexjoco-openpi"
    wandb_mode: Literal["online", "offline"] = "online"
    conda_env: str = "openpi"
    mem_fraction: float = 0.9
    nccl_p2p_disable: bool = False
    dry_run: bool = False


PROXY_ENV_VARS = [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
]


def session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        check=False,
        capture_output=True,
    )
    return result.returncode == 0


def build_command(
    config_name: str,
    gpu_id: str,
    wandb_mode: Literal["online", "offline"],
    nccl_p2p_disable: bool,  # noqa: FBT001
    mem_fraction: float,
    conda_env: str,
    wandb_project: str,
) -> str:
    env_prefix = []
    if wandb_mode == "offline":
        env_prefix.append("WANDB_MODE=offline")

    if nccl_p2p_disable:
        env_prefix.append("NCCL_P2P_DISABLE=1")

    env_prefix.extend(
        [
            f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}",
            f"CUDA_VISIBLE_DEVICES={gpu_id}",
        ]
    )
    env_str = " ".join(env_prefix)
    unset_proxy_cmd = f"unset {' '.join(PROXY_ENV_VARS)}"

    return (
        f"{unset_proxy_cmd} && "
        f"conda activate {conda_env} && "
        f"{env_str} python scripts/train.py {config_name} --project-name {wandb_project}"
    )


def main(args: Args) -> None:
    # print train information
    assert len(args.config_names) == len(args.gpus), "Number of config names must match number of GPUs"
    for config_name, gpu_id in zip(args.config_names, args.gpus, strict=True):
        print(f"Config: {config_name}, GPU: {gpu_id}")
    print(f"WandB Project: {args.wandb_project}, WandB Mode: {args.wandb_mode}")
    print(f"Conda Environment: {args.conda_env}, Memory Fraction: {args.mem_fraction}")

    for config_name, gpu_id in zip(args.config_names, args.gpus, strict=True):
        session_name = f"cuda{gpu_id.replace(',', '')}"

        command = build_command(
            config_name=config_name,
            gpu_id=gpu_id,
            wandb_mode=args.wandb_mode,
            nccl_p2p_disable=args.nccl_p2p_disable,
            mem_fraction=args.mem_fraction,
            conda_env=args.conda_env,
            wandb_project=args.wandb_project,
        )
        if args.dry_run:
            print(f"[dry-run] session {session_name}: {command}")
            continue

        if session_exists(session_name):
            print(f"skip existing session: {session_name}")
            continue

        subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name],
            check=True,
        )
        subprocess.run(
            ["tmux", "send-keys", "-t", session_name, command, "C-m"],
            check=True,
        )
        print(f"started {session_name}: {command}")


if __name__ == "__main__":
    main(tyro.cli(Args))
