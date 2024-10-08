from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Optional

from src.utils.randomizer import RandomizerI, UniformRandomizerI, NormalRandomizerI, DefaultUniformRandomizer, \
    DefaultNormalRandomizer


@dataclass
class BanditsConfig:
    """Base configuration for all bandit environments"""
    num_bandits: int = 10               # Number of possible actions
    max_steps: int = 1000               # The length of an episode
    seed: Optional[int] = None          # Set random state


BaseConfigType = TypeVar("BaseConfigType", bound=BanditsConfig)


class Bandits(ABC):
    """Base class for all bandit environments"""

    def __init__(self, config: BaseConfigType, randomizer: RandomizerI):
        self.config: BaseConfigType = config
        self.step_num: int = 0
        self.rng = randomizer(self.config.seed)

    @property
    def num_bandits(self) -> int:
        return self.config.num_bandits

    def reset(self) -> None:
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


BanditsType = TypeVar("BanditsType", bound=Bandits)


@dataclass
class FixedBanditsConfig(BanditsConfig):
    hit_reward_value: int = 10          # The winning reward for all bandits
    miss_reward_value: int = -1         # The losing reward for all bandits


class FixedValueBandits(Bandits):
    """Bandit environment where each bandit has a fixed probability of
    yielding a winning or losing reward at any time step.
    The rewards returned are fixed and the same for all bandits."""

    config: FixedBanditsConfig
    rng: UniformRandomizerI

    def __init__(self, config: FixedBanditsConfig, randomizer: UniformRandomizerI):
        super().__init__(config, randomizer)
        self.probabilities = self.rng.uniform(0.001, 1.0, self.num_bandits)

    def get_reward(self, action: int) -> float:
        """Get reward from bandit #{action}"""
        action_roll = self.rng.uniform(0.001, 1)
        if action_roll >= self.probabilities[action]:
            return self.config.hit_reward_value
        else:
            return self.config.miss_reward_value


@dataclass
class GaussianBanditsConfig(BanditsConfig):
    global_reward_mean: float = 0.      # The mean reward for all bandits
    global_reward_sigma: float = 1.     # The deviations of the mean reward for all bandits
    reward_sigma: float = 1.            # The deviations of the reward on each action


class GaussianValueBandits(Bandits):
    """Bandit environment where each bandit yields a reward each time
    is selected. Each bandit has a mean reward that its fixed, and
    generated when the environment is created, from a normal distribution.
    When a bandit is selected the reward is generated from a normal
    distribution, with mu equal to the bandits fixed mean reward."""

    config: GaussianBanditsConfig
    rng: NormalRandomizerI

    def __init__(self, config: GaussianBanditsConfig, randomizer: NormalRandomizerI):
        super().__init__(config, randomizer)
        self.rewards = self.rng.normal(loc=self.config.global_reward_mean,
                                       scale=self.config.global_reward_sigma,
                                       size=self.num_bandits)

    def get_reward(self, action: int) -> float:
        """Get reward from bandit #{action}"""
        reward = self.rng.normal(loc=self.rewards[action],
                                 scale=self.config.reward_sigma)
        return reward


def main():
    print("Fixed bandits")
    f_bandits = FixedValueBandits(FixedBanditsConfig(max_steps=20),
                                  DefaultUniformRandomizer())
    
    done = False
    while not done:
        for i in range(f_bandits.num_bandits):
            _, reward, done = f_bandits.step(i)
            print(reward)

    print("Gaussian bandits")
    g_bandits = GaussianValueBandits(GaussianBanditsConfig(max_steps=20),
                                     DefaultNormalRandomizer())

    done = False
    while not done:
        for i in range(g_bandits.num_bandits):
            _, reward, done = g_bandits.step(i)
            print(reward)
    pass


if __name__ == '__main__':
    main()
