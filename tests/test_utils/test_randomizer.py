class RandomizerSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, seed: int) -> 'RandomizerSpy':
        self.calls.append(seed)
        return self


class UniformRandomizerSpy(RandomizerSpy):

    uniform_calls: list[tuple[float, float, int | None]] = []
    single_values_calls: int = 0

    def uniform(self, low: float, high: float, size: int | None = None) -> float | list[float]:
        self.uniform_calls.append((low, high, size))

        if size is None or size <= 1:
            self.single_values_calls += 1
            # Even calls
            if self.single_values_calls % 2 == 0:
                return 0.8

            return 0.1

        return [0.2] * size


class NormalRandomizerSpy(RandomizerSpy):

    normal_calls: list[tuple[float, float, int | None]] = []
    single_values_calls: int = 0

    def normal(self, loc: float, scale: float, size: int | None = None) -> float | list[float]:
        self.normal_calls.append((loc, scale, size))
        if size is None or size <= 1:
            self.single_values_calls += 1
            # Even calls
            if self.single_values_calls % 2 == 0:
                return 0.8

            return 0.1

        return [0.2] * size
