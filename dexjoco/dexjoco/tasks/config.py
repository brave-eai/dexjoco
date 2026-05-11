from abc import ABC, abstractmethod

class TaskConfigBase(ABC):
    """Base config for task environments."""

    proprio_keys = None

    @abstractmethod
    def get_environment(self, policy_mode=False, render_mode="human", randomize=False, **kwargs):
        """Create the task environment.

        When ``policy_mode`` is true, implementations are expected to attach the
        policy action wrapper instead of the teleoperation wrapper.
        """
        pass

    @abstractmethod
    def process_demos(self, demo):
        pass
