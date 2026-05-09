from abc import abstractmethod

class TaskConfigBase:
    """Base config for simulated teleoperation data collection."""

    proprio_keys = None

    @abstractmethod
    def get_environment(self, fake_env=False, render_mode="human", randomize=False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
