from abc import ABC, abstractmethod
import torch

class LeagueAgent(ABC):
    """
    The Universal Interface.
    Any class inheriting from this can be loaded by the League Manager.
    """
    def __init__(self, name: str, obs_shape: tuple, action_dim: int, device: str):
        self.name = name
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.steps = 0

    @abstractmethod
    def get_action(self, obs, eval_mode=False):
        """
        Input: Observation (numpy array)
        Output: (action, metadata_dict)
        """
        pass

    @abstractmethod
    def store(self, obs, action, reward, next_obs, done, **kwargs):
        """
        Stores experience. 
        **kwargs is critical: allows PPO to pass 'log_prob' while DQN ignores it.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Internal check: "Do I have enough data to train?"
        If yes, run optimization loop.
        """
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass