from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class BanditsConfig:
    num_bandits: int = 10
    max_steps: int = 1000
    reward_value: int = 10


class Bandits(ABC):

    def __init__(self, config: BanditsConfig):
        self.config = config
        self.step_num = 0

    @property
    def num_bandits(self):
        return self.config.num_bandits

    def reset(self):
        self.step_num = 0

    @abstractmethod
    def get_reward(self, action: int) -> float:
        """Get reward from bandit #{action}"""
    
    def step(self, action: int) -> Tuple[List[float], float, bool]:
        reward = self.get_reward(action)
        self.step_num += 1
        done = self.step_num >= self.config.max_steps
        return [], reward, done


class FixedValueBandits(Bandits):

    def __init__(self, config: BanditsConfig):
        super().__init__(config)
        self.probabilities = np.random.uniform(0.001, 1, self.num_bandits)

    def get_reward(self, action: int) -> float:
        action_roll = np.random.uniform(0.001, 1)
        if action_roll >= self.probabilities[action]:
            return self.config.reward_value
        else:
            return 0.


class GaussianValueBandits(Bandits):

    def __init__(self, config: BanditsConfig):
        super().__init__(config)
        self.rewards = np.random.normal(size=self.num_bandits)

    def get_reward(self, action: int) -> float:
        reward = np.random.normal(self.rewards[action])
        return reward


def main():
    print("Fixed bandits")
    f_bandits = FixedValueBandits(BanditsConfig(max_steps=20))
    
    done = False
    while not done:
        for i in range(f_bandits.num_bandits):
            _, reward, done = f_bandits.step(i)
            print(reward)

    print("Gaussian bandits")
    g_bandits = GaussianValueBandits(BanditsConfig(max_steps=20))

    done = False
    while not done:
        for i in range(g_bandits.num_bandits):
            _, reward, done = g_bandits.step(i)
            print(reward)
    pass


if __name__ == '__main__':
    main()
