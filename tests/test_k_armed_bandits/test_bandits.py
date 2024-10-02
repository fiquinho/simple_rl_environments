from pytest import fixture

from src.k_armed_bandits.bandits import (Bandits, BanditsConfig, FixedBanditsConfig, FixedValueBandits,
                                         GaussianBanditsConfig, GaussianValueBandits)
from tests.test_utils.test_randomizer import RandomizerSpy, UniformRandomizerSpy, NormalRandomizerSpy


@fixture
def bandits_config() -> BanditsConfig:
    return BanditsConfig(num_bandits=3, max_steps=5, seed=0)


def test_bandits(bandits_config):
    randomizer = RandomizerSpy()
    bandits = BanditsTestImplementation(config=bandits_config, randomizer=randomizer)

    assert bandits.num_bandits == 3
    assert randomizer.calls == [0]
    assert bandits.step_num == 0

    bandits.step_num = 3
    bandits.reset()
    assert bandits.step_num == 0
    assert randomizer.calls == [0]

    environment_sate, reward, is_episode_done = bandits.step(1)
    assert bandits.step_num == 1
    assert bandits.get_reward_calls == [1]
    assert environment_sate == []
    assert reward == 1.
    assert not is_episode_done

    bandits.step_num = 4
    environment_sate, reward, is_episode_done = bandits.step(2)
    assert bandits.step_num == 5
    assert bandits.get_reward_calls == [1, 2]
    assert environment_sate == []
    assert reward == 2.
    assert is_episode_done


@fixture
def fixed_bandits_config() -> FixedBanditsConfig:
    return FixedBanditsConfig(num_bandits=3, max_steps=5, seed=0, hit_reward_value=10,
                              miss_reward_value=-1)


def test_fixed_value_bandits(fixed_bandits_config):
    randomizer = UniformRandomizerSpy()
    bandits = FixedValueBandits(config=fixed_bandits_config, randomizer=randomizer)

    assert bandits.probabilities == [0.2, 0.2, 0.2]
    assert randomizer.uniform_calls == [(0.001, 1., 3)]

    reward = bandits.get_reward(0)
    assert reward == -1.
    assert randomizer.uniform_calls[-1] == (0.001, 1., None)

    reward = bandits.get_reward(1)
    assert reward == 10.
    assert randomizer.uniform_calls[-1] == (0.001, 1., None)


@fixture
def gaussian_bandits_config() -> GaussianBanditsConfig:
    return GaussianBanditsConfig(num_bandits=3, max_steps=5, seed=0, global_reward_mean=0.,
                                 global_reward_sigma=1., reward_sigma=-1.)


def test_gaussian_value_bandits(gaussian_bandits_config):
    randomizer = NormalRandomizerSpy()
    bandits = GaussianValueBandits(config=gaussian_bandits_config, randomizer=randomizer)

    assert bandits.rewards == [0.2, 0.2, 0.2]
    assert randomizer.normal_calls == [(0., 1., 3)]

    reward = bandits.get_reward(0)
    assert reward == 0.1
    assert randomizer.normal_calls[-1] == (0.2, -1., None)

    reward = bandits.get_reward(1)
    assert reward == 0.8
    assert randomizer.normal_calls[-1] == (0.2, -1., None)


class BanditsTestImplementation(Bandits):

    get_reward_calls: list[int] = []

    def get_reward(self, action: int) -> float:
        self.get_reward_calls.append(action)
        if action == 1:
            return 1.0
        elif action == 2:
            return 2.0
        else:
            return 0.0
