from dataclasses import dataclass
from typing import Literal

from tqdm import tqdm


@dataclass(frozen=True)
class ProgressMsg:
    dataset_name: str
    episode_name: str


@dataclass(frozen=True)
class InitMsg:
    dataset_name: str
    total_episodes: int


@dataclass(frozen=True)
class DoneMsg:
    dataset_name: str
    statistics_msg: str = "Dataset processing completed."


@dataclass(frozen=True)
class ErrorMsg:
    dataset_name: str
    error: str
    episode_name: str | None = None


WorkerMsg = InitMsg | ProgressMsg | DoneMsg | ErrorMsg


@dataclass
class DatasetProgressState:
    total_episodes: int = 0
    finished_episodes: int = 0
    status: Literal["running", "done", "failed"] = "running"
    done_msg: str | None = None
    last_episode: str | None = None
    last_error: str | None = None


class ProgressBar:
    def __init__(self, idx: int, dataset_name: str):
        """Create a progress bar for one dataset.

        Args:
            idx: Display row index used by ``tqdm``.
            dataset_name: Dataset identifier accepted by this progress bar.

        Returns:
            None.
        """
        self.state = DatasetProgressState()
        self.dataset_name = dataset_name
        self._bar = tqdm(
            desc=dataset_name,
            unit="episode",
            position=idx,
            leave=True,
        )

    def update(self, msg: WorkerMsg):
        """Apply a worker message and refresh the rendered progress bar.

        Args:
            msg: Progress, initialization, completion, or error message emitted
                by a worker process for this dataset.

        Returns:
            None.
        """
        self._apply_message(msg)
        self._bar.n = self.state.finished_episodes

        if self._bar.total != self.state.total_episodes:
            self._bar.total = self.state.total_episodes

        last_episode = (
            self.state.last_episode if self.state.last_episode is not None else "-"
        )
        self._bar.set_postfix_str(
            f"{self.state.status} | {last_episode}", refresh=False
        )

        self._bar.refresh()

    def _apply_message(self, msg: WorkerMsg):
        """Update the in-memory progress state from a worker message.

        Args:
            msg: Worker message to apply to this progress state.

        Returns:
            None.

        Raises:
            AssertionError: If the message belongs to another dataset.
            ValueError: If episode counts are invalid or overflow the total.
            TypeError: If the message type is unsupported.
        """
        assert msg.dataset_name == self.dataset_name, (
            "Message dataset_name does not match ProgressBar dataset_name"
        )
        match msg:
            case InitMsg(dataset_name=dataset_name, total_episodes=total):
                if total < 0:
                    raise ValueError(
                        f"dataset {dataset_name} has invalid total_episodes: {total}"
                    )
                self.state.total_episodes = total

            case ProgressMsg(dataset_name=dataset_name, episode_name=ep):
                self.state.finished_episodes += 1
                if self.state.finished_episodes > self.state.total_episodes:
                    raise ValueError(
                        f"dataset {dataset_name} overflow: "
                        f"{self.state.finished_episodes} > {self.state.total_episodes}"
                    )
                self.state.last_episode = ep

            case DoneMsg(dataset_name=dataset_name, statistics_msg=stat_msg):
                self.state.done_msg = stat_msg
                self.state.status = "done"

            case ErrorMsg(dataset_name=dataset_name, error=err, episode_name=ep):
                self.state.status = "failed"
                self.state.last_error = err
                self.state.last_episode = ep

            case _:
                raise TypeError(f"Unsupported message type: {type(msg).__name__}")

    def close(self):
        """Close the underlying progress bar.

        Args:
            None.

        Returns:
            None.
        """
        self._bar.close()
