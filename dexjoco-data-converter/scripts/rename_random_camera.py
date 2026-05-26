"""
This file renames "camera_*.mp4" to "random_camera.mp4".
"""


from pathlib import Path


def check_one_dataset(dataset_dir: Path):
    all_video_dirs: list[Path] = []
    for episode_dir in dataset_dir.iterdir():
        assert episode_dir.is_dir()
        video_dir = episode_dir / "videos"
        all_video_dirs.append(video_dir)

    camera_counts: dict[Path, int] = {}
    for video_dir in all_video_dirs:
        random_camera_videos = list(video_dir.glob("camera_*.mp4"))
        camera_counts[video_dir] = len(random_camera_videos)
        if (video_dir / "random_camera.mp4").exists():
            camera_counts[video_dir] += 1

    if any(count != 1 for count in camera_counts.values()):
        return False
    else:
        return True


def rename_one_dataset(dataset_dir: Path):
    for episode_dir in dataset_dir.iterdir():
        assert episode_dir.is_dir()
        video_dir = episode_dir / "videos"
        random_camera_videos = list(video_dir.glob("camera_*.mp4"))
        dst_video = video_dir / "random_camera.mp4"

        if len(random_camera_videos) + int(dst_video.exists()) != 1:
            raise ValueError(f"{video_dir}: expected exactly one random camera video or existing random_camera.mp4, found {len(random_camera_videos)} camera_*.mp4 and random_camera.mp4 exists={dst_video.exists()}")
        
        if dst_video.exists():
            continue
        else:
            src_video = random_camera_videos[0]
            src_video.rename(dst_video)


def batch_rename_random_camera(datasets_dir: Path):
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    dataset_dirs.sort(key=lambda d: d.name)
    for dataset_dir in dataset_dirs:
        if not check_one_dataset(dataset_dir):
            print(f"{dataset_dir}: Skipping because not all video dirs have exactly one random camera")
            continue
        rename_one_dataset(dataset_dir)
        print(f"{dataset_dir}: Renamed random camera videos to random_camera.mp4")


if __name__ == "__main__":
    import tyro

    tyro.cli(batch_rename_random_camera)


"""
python scripts/rename_random_camera.py --datasets-dir /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/replay_random_data
"""
