from abc import ABC, abstractmethod

class TaskConfigBase(ABC):
    """Base config for task environments."""

    proprio_keys = None

    @abstractmethod
    def get_environment(self, policy_mode=False, render_mode="human", randomize=False, **kwargs):
        pass

    @abstractmethod
    def process_demos(self, demo):
        pass
