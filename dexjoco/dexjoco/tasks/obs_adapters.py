from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space


class DexjocoObsAdapter(gym.ObservationWrapper):
    """Adapt raw task observations into the flattened Dexjoco collection format."""

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        ordered_spaces = OrderedDict(
            (key, self.env.observation_space["state"][key]) for key in self.proprio_keys
        )
        self.proprio_space = gym.spaces.Dict(ordered_spaces)

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space.get("images", {})),
            }
        )

    def observation(self, obs):
        proprio_values = [np.asarray(obs["state"][key]).ravel() for key in self.proprio_keys]
        if proprio_values:
            flat_state = np.concatenate(proprio_values, axis=0).astype(np.float32, copy=False)
        else:
            flat_state = np.zeros((0,), dtype=np.float32)
        return {"state": flat_state, **obs.get("images", {})}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info
