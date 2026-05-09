"""Time-major Zarr storage for recorded Dexjoco episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numcodecs
import numpy as np
import zarr


def _as_time_major_array(name: str, value) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        raise ValueError(f"{name} must have a time dimension.")
    if array.dtype == object:
        raise TypeError(f"{name} has object dtype; all episode fields must have fixed shapes.")
    return np.ascontiguousarray(array)


def _chunk_shape(shape: tuple[int, ...], dtype: np.dtype, target_bytes: int = 2_000_000) -> tuple[int, ...]:
    itemsize = np.dtype(dtype).itemsize
    trailing = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
    bytes_per_step = max(itemsize * trailing, 1)
    chunk_length = max(1, min(shape[0], target_bytes // bytes_per_step))
    return (chunk_length, *shape[1:])


def _resolve_compressor(compressors):
    if compressors in (None, False, "none"):
        return None
    if compressors == "disk":
        return numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    raise ValueError(f"Unsupported compressor setting: {compressors!r}")


@dataclass
class ZarrEpisodeStore:
    """Append-only storage for complete recorded episodes."""

    root: zarr.Group

    @classmethod
    def create_empty(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        root.require_group("data")
        meta_group = root.require_group("meta")
        if "episode_ends" not in meta_group:
            meta_group.array(
                "episode_ends",
                data=np.zeros((0,), dtype=np.int64),
                chunks=(1024,),
                compressor=None,
                overwrite=False,
            )
        return cls(root=root)

    @property
    def data(self) -> zarr.Group:
        return self.root["data"]

    @property
    def meta(self) -> zarr.Group:
        return self.root["meta"]

    @property
    def episode_ends(self) -> zarr.Array:
        return self.meta["episode_ends"]

    @property
    def total_steps(self) -> int:
        if self.episode_ends.shape[0] == 0:
            return 0
        return int(self.episode_ends[-1])

    def append_episode(self, episode_data: Dict[str, np.ndarray], compressors="disk") -> None:
        if not episode_data:
            raise ValueError("episode_data must not be empty.")

        arrays = {name: _as_time_major_array(name, value) for name, value in episode_data.items()}
        lengths = {array.shape[0] for array in arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"All episode fields must have the same length, got {sorted(lengths)}.")

        episode_length = lengths.pop()
        if episode_length <= 0:
            raise ValueError("Episode length must be positive.")

        compressor = _resolve_compressor(compressors)
        start = self.total_steps
        end = start + episode_length

        existing_keys = set(self.data.keys())
        new_keys = set(arrays.keys())
        if existing_keys and existing_keys != new_keys:
            raise ValueError(
                f"Episode fields changed across recordings. Existing keys: {sorted(existing_keys)}, "
                f"new keys: {sorted(new_keys)}."
            )

        for name, array in arrays.items():
            if name not in self.data:
                self.data.create_dataset(
                    name=name,
                    data=array,
                    shape=array.shape,
                    chunks=_chunk_shape(array.shape, array.dtype),
                    dtype=array.dtype,
                    compressor=compressor,
                    overwrite=False,
                )
                continue

            dataset = self.data[name]
            if dataset.shape[1:] != array.shape[1:]:
                raise ValueError(f"{name} shape changed from {dataset.shape[1:]} to {array.shape[1:]}.")
            if dataset.dtype != array.dtype:
                raise ValueError(f"{name} dtype changed from {dataset.dtype} to {array.dtype}.")

            dataset.resize((end, *dataset.shape[1:]))
            dataset[start:end] = array

        episode_ends = self.episode_ends
        new_episode_count = episode_ends.shape[0] + 1
        episode_ends.resize((new_episode_count,))
        episode_ends[-1] = end
