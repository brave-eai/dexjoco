"""
This file checks if any episode has multiple "camera_*.mp4" videos.
"""


from pathlib import Path

def check_one_dataset(dataset_dir: Path, verbose: bool = False):
    all_video_dirs: list[Path] = []
    for episode_dir in dataset_dir.iterdir():
        assert episode_dir.is_dir()
        video_dir = episode_dir / "videos"
        all_video_dirs.append(video_dir)

    camera_counts: dict[Path, int] = {}
    for video_dir in all_video_dirs:
        random_camera_videos = list(video_dir.glob("camera_*.mp4"))
        camera_counts[video_dir] = len(random_camera_videos)

    if any(count != 1 for count in camera_counts.values()):
        print(f"{dataset_dir}: Found video dirs with non-single random cameras")
    else:
        print(f"{dataset_dir}: All video dirs have exactly one random camera")

    if verbose:
        for video_dir, count in camera_counts.items():
            if count != 1:
                print(f"  {video_dir}: {count} random camera videos")


def batch_check_random_camera(datasets_dir: Path, verbose: bool = False):
    dataset_dirs = [d for d in datasets_dir.iterdir() if d.is_dir()]
    dataset_dirs.sort(key=lambda d: d.name)
    for dataset_dir in dataset_dirs:
        check_one_dataset(dataset_dir, verbose=verbose)


if __name__ == "__main__":
    import tyro

    tyro.cli(batch_check_random_camera)


"""
python scripts/check_random_camera.py --datasets-dir /data/weizhi_zhao/dexjoco/dexjoco_raw_datasets/replay_random_data
"""
