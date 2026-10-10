import multiprocessing as mp
import queue
from multiprocessing.synchronize import Event as EventClass
from pathlib import Path

import av


def decode_video(video_path: Path, stop_event: EventClass, frame_queue: mp.Queue):
    def put(item):
        while not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.1)
                break
            except queue.Full:
                continue

    try:
        with av.open(str(video_path)) as container:
            for idx, frame in enumerate(container.decode(video=0)):
                if stop_event.is_set():
                    break
                put((frame, idx))
            put(None)
    except Exception as e:  # noqa: BLE001
        print(e)
        put(e)
