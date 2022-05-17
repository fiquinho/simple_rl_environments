from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Protocol
import numpy as np


class ConfigProtocol(Protocol):
    """Protocol for all bandits configurations"""
    num_bandits: int
    max_steps: int


@dataclass
class BanditsConfig:
    """Base configuration for all bandit environments"""
    num_bandits: int = 10               # Number of possible actions
    max_steps: int = 1000               # The length of an episode


@dataclass
class FixedBanditsConfig(BanditsConfig):
    reward_value: int = 10              # The yielded reward from a bandit


@dataclass
class GaussianBanditsConfig(BanditsConfig):
    global_reward_mean: float = 0.      # The mean reward fro all bandits
    global_reward_sigma: float = 1.     # The deviations of the mean reward for all bandits
    reward_sigma: float = 1.            # The deviations of the reward on each action


class Bandits(ABC):
    """Base class for all bandit environments"""

    def __init__(self, config: ConfigProtocol):
        self.config = config
        self.step_num = 0

    @property
    def num_bandits(self):
        return self.config.num_bandits

    def reset(self):
        # Reset the environment to start a new episode
        self.step_num = 0

    @abstractmethod
    def get_reward(self, action: int) -> float:
        """Get reward from bandit #{action}"""
    
    def step(self, action: int) -> Tuple[List[float], float, bool]:
        """Take a single action in the environment

        :param action: The bandit to use on this time step
        :return: (environment_sate, reward, is_episode_done)
        """
        reward = self.get_reward(action)
        self.step_num += 1
        done = self.step_num >= self.config.max_steps
        return [], reward, done


class FixedValueBandits(Bandits):
    """Bandit environment where each bandit has a fixed probability of
    yielding a reward at any time step. The reward returned is fixed and
    the same for all bandits."""

    config: FixedBanditsConfig

    def __init__(self, config: FixedBanditsConfig):
        super().__init__(config)
        self.probabilities = np.random.uniform(0.001, 1, self.num_bandits)

    def get_reward(self, action: int) -> float:
        """Get reward from bandit #{action}"""
        action_roll = np.random.uniform(0.001, 1)
        if action_roll >= self.probabilities[action]:
            return self.config.reward_value
        else:
            return 0.


class GaussianValueBandits(Bandits):
    """Bandit environment where each bandit yields a reward each time
    is selected. Each bandit has a mean reward that its fixed, and
    generated when the environment is created, from a normal distribution.
    When a bandit is selected the reward is generated from a normal
    distribution, with mu equal to the bandits fixed mean reward."""

    config: GaussianBanditsConfig

    def __init__(self, config: GaussianBanditsConfig):
        super().__init__(config)
        self.rewards = np.random.normal(loc=self.config.global_reward_mean,
                                        scale=self.config.global_reward_sigma,
                                        size=self.num_bandits)

    def get_reward(self, action: int) -> float:
        """Get reward from bandit #{action}"""
        reward = np.random.normal(loc=self.rewards[action],
                                  scale=self.config.reward_sigma)
        return reward


def main():
    print("Fixed bandits")
    f_bandits = FixedValueBandits(FixedBanditsConfig(max_steps=20))
    
    done = False
    while not done:
        for i in range(f_bandits.num_bandits):
            _, reward, done = f_bandits.step(i)
            print(reward)

    print("Gaussian bandits")
    g_bandits = GaussianValueBandits(GaussianBanditsConfig(max_steps=20))

    done = False
    while not done:
        for i in range(g_bandits.num_bandits):
            _, reward, done = g_bandits.step(i)
            print(reward)
    pass


if __name__ == '__main__':
    main()
