import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path
from typing import Any

import av

_IMAGE_QUEUES: dict[str, mp.Queue[Any]] = {}
_STOP_EVENT: EventClass


def initialize_decoder(
    image_queues: dict[str, mp.Queue[Any]], stop_event: EventClass
) -> None:
    """Receive shared queues and the stop event when each worker starts."""
    global _IMAGE_QUEUES, _STOP_EVENT
    _IMAGE_QUEUES = image_queues
    _STOP_EVENT = stop_event

    for q in image_queues.values():
        q.cancel_join_thread()


def decode_video(video_path: Path, camera_name: str):
    image_queue = _IMAGE_QUEUES[camera_name]
    stop_event = _STOP_EVENT

    def put(item):
        while not stop_event.is_set():
            try:
                image_queue.put(item, timeout=0.1)
                break
            except queue.Full:
                continue

    try:
        with av.open(str(video_path)) as container:
            for idx, frame in enumerate(container.decode(video=0)):
                if stop_event.is_set():
                    break
                put((frame.to_ndarray(format="rgb24"), idx))
            put(None)
    except Exception as e:  # noqa: BLE001
        print(e)
        put(e)
