import tyro

from dexjoco_data_converter.to_lerobot.batch_merge_episode_lerobot import (
    batch_merge_episode as batch_lerobot,
)
from dexjoco_data_converter.to_lerobot.merge_datasets_lerobot import (
    merge_datasets as merge_lerobot,
)
from dexjoco_data_converter.to_lerobot.merge_episode_lerobot import (
    merge_episode as single_lerobot,
)
from dexjoco_data_converter.to_zarr.batch_merge_episode_zarr import (
    batch_merge_episode as batch_zarr,
)
from dexjoco_data_converter.to_zarr.merge_datasets_zarr import (
    merge_datasets as merge_zarr,
)
from dexjoco_data_converter.to_zarr.merge_episode_zarr import (
    merge_episode as single_zarr,
)


def single_lerobot_cli() -> None:
    tyro.cli(single_lerobot)


def single_zarr_cli() -> None:
    tyro.cli(single_zarr)


def batch_lerobot_cli() -> None:
    tyro.cli(batch_lerobot)


def batch_zarr_cli() -> None:
    tyro.cli(batch_zarr)


def merge_lerobot_cli() -> None:
    tyro.cli(merge_lerobot)


def merge_zarr_cli() -> None:
    tyro.cli(merge_zarr)
