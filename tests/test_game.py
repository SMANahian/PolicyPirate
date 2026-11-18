import itertools
from collections import Counter

import numpy as np
import pytest

from policypirate.game import BattleshipEnv, GameConfig, random_fleet, ship_placements


def play_random_game(env, seed):
    """Fire at random legal cells until the game ends; return the number of shots."""
    env.reset(seed=seed)
    done = False
    while not done:
        cell = env.np_random.choice(np.flatnonzero(env.action_masks()))
        _, _, done, _, info = env.step(cell)
    return info["shots"]


def test_placement_count():
    # A length-5 ship on 10x10: 6 starting columns x 10 rows, horizontally and vertically.
    assert len(ship_placements(GameConfig(), 5)) == 120


def test_fleets_never_overlap():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        cells = [cell for ship in random_fleet(GameConfig(), rng) for cell in ship]
        assert len(cells) == len(set(cells)) == 17


def test_every_fleet_equally_likely():
    # On a tiny board we can list every legal fleet, then check each one comes up about equally often.
    cfg = GameConfig(height=3, width=3, fleet=(2, 2))
    options = [ship_placements(cfg, length) for length in cfg.fleet]
    legal = [fleet for fleet in itertools.product(*options) if not set(fleet[0]) & set(fleet[1])]
    rng = np.random.default_rng(0)
    counts = Counter(tuple(random_fleet(cfg, rng)) for _ in range(500 * len(legal)))
    assert set(counts) == set(legal)  # only legal fleets, and all of them
    assert all(400 < n < 600 for n in counts.values())  # about 500 each


def test_shots_rewards_and_sinking():
    for mode, hit_reward in [("shot", -1), ("miss", 0)]:
        env = BattleshipEnv(reward=mode)
        env.reset(seed=0)
        water = int(np.flatnonzero(env.ship_at == -1)[0])
        destroyer = env.fleet[4]  # the length-2 ship

        _, reward, _, _, _ = env.step(water)
        assert reward == -1
        _, reward, _, _, _ = env.step(destroyer[0])
        assert reward == hit_reward
        obs, reward, done, _, _ = env.step(destroyer[1])
        assert reward == hit_reward and not done
        assert obs[3 + 4].ravel()[destroyer[1]] == 1 and obs[3 + 4].sum() == 1  # sunk here, nowhere else

        with pytest.raises(ValueError):  # firing at the same cell twice
            env.step(water)


def test_random_play_matches_exact_average():
    # Random shooting ends when the last of 17 ship cells (out of 100) is hit,
    # which takes 17 * 101 / 18 = 95.4 shots on average.
    shots = [play_random_game(BattleshipEnv(), seed) for seed in range(1000)]
    assert abs(np.mean(shots) - 17 * 101 / 18) < 0.6
