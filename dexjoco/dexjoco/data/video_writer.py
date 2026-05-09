"""Simple H.264/MP4 writing utilities for Dexjoco recordings."""

from __future__ import annotations

from dataclasses import dataclass, field

import av
import numpy as np


@dataclass
class Mp4VideoWriter:
    fps: int
    codec: str
    input_pix_fmt: str
    output_pix_fmt: str = "yuv420p"
    codec_options: dict[str, str] = field(default_factory=dict)
    thread_type: str | None = None
    thread_count: int | None = None

    def __post_init__(self):
        self._container = None
        self._stream = None
        self._shape = None

    @classmethod
    def create_h264(
        cls,
        fps,
        codec="h264",
        input_pix_fmt="rgb24",
        output_pix_fmt="yuv420p",
        crf=18,
        profile="high",
        thread_type=None,
        thread_count=None,
        **_,
    ):
        codec_options = {
            "crf": str(crf),
            "profile": profile,
        }
        return cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            output_pix_fmt=output_pix_fmt,
            codec_options=codec_options,
            thread_type=thread_type,
            thread_count=thread_count,
        )

    def start(self, file_path):
        self.stop()
        self._container = av.open(file_path, mode="w")
        self._stream = self._container.add_stream(self.codec, rate=self.fps)
        self._stream.pix_fmt = self.output_pix_fmt
        if self.thread_type is not None:
            self._stream.thread_type = self.thread_type
        if self.thread_count is not None:
            self._stream.thread_count = self.thread_count
        if self.codec_options:
            self._stream.options = dict(self.codec_options)
        self._shape = None

    def write_frame(self, image: np.ndarray):
        if self._stream is None or self._container is None:
            raise RuntimeError("start() must be called before write_frame().")

        frame_array = np.asarray(image)
        if frame_array.ndim != 3 or frame_array.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {frame_array.shape}.")
        if frame_array.dtype != np.uint8:
            raise TypeError(f"Expected uint8 image, got dtype {frame_array.dtype}.")

        frame_array = np.ascontiguousarray(frame_array)
        if self._shape is None:
            self._shape = frame_array.shape
            height, width, _ = frame_array.shape
            self._stream.width = width
            self._stream.height = height
        elif frame_array.shape != self._shape:
            raise ValueError(f"Frame shape changed from {self._shape} to {frame_array.shape}.")

        video_frame = av.VideoFrame.from_ndarray(frame_array, format=self.input_pix_fmt)
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def stop(self):
        if self._stream is not None and self._container is not None:
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
        self._container = None
        self._stream = None
        self._shape = None

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass
