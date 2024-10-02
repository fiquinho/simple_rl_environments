from typing import Protocol

import numpy as np


class RandomizerI(Protocol):

    def __call__(self, seed: int | None) -> 'RandomizerI':
        ...


class UniformRandomizerI(RandomizerI, Protocol):

    def uniform(self, low: float, high: float, size: int | None = None) -> float | list[float]:
        ...


class NormalRandomizerI(RandomizerI, Protocol):

    def normal(self, loc: float, scale: float, size: int | None = None) -> float | list[float]:
        ...


class DefaultUniformRandomizer:

    rng: np.random.Generator = np.random.default_rng()

    def __call__(self, seed: int | None) -> 'RandomizerI':
        self.rgn = np.random.default_rng(seed)
        return self

    def uniform(self, low: float, high: float, size: int | None = None) -> float | list[float]:
        return self.rgn.uniform(low, high, size)


class DefaultNormalRandomizer:

    rng: np.random.Generator = np.random.default_rng()

    def __call__(self, seed: int | None) -> 'RandomizerI':
        self.rgn = np.random.default_rng(seed)
        return self

    def normal(self, loc: float, scale: float, size: int | None = None) -> float | list[float]:
        return self.rgn.normal(loc, scale, size)
