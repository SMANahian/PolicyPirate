"""The Battleship game: its rules and the environment the agent plays in."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class GameConfig:
    """Board size and ship lengths. GameConfig() is the classic game.

    Ships are straight (horizontal or vertical) and may touch but not overlap.
    These are the usual rules, the ones other bots and published results assume.
    """

    height: int = 10
    width: int = 10
    fleet: tuple[int, ...] = (5, 4, 3, 3, 2)

    @property
    def num_cells(self) -> int:
        return self.height * self.width


def ship_placements(cfg: GameConfig, length: int) -> list[tuple[int, ...]]:
    """Every position a ship of this length can occupy, each as a tuple of cell numbers.

    Cells are numbered row by row: cell = row * width + col, so 0..99 on the classic board.
    """
    placements = []
    for row in range(cfg.height):  # horizontal
        for col in range(cfg.width - length + 1):
            placements.append(tuple(row * cfg.width + col + k for k in range(length)))
    for row in range(cfg.height - length + 1):  # vertical
        for col in range(cfg.width):
            placements.append(tuple((row + k) * cfg.width + col for k in range(length)))
    return placements


def random_fleet(cfg: GameConfig, rng: np.random.Generator) -> list[tuple[int, ...]]:
    """A random fleet (one placement per ship) with every legal fleet equally likely.

    Each ship gets an independent random placement; if any two ships overlap, the whole
    fleet is thrown away and redrawn.
    """
    options = [ship_placements(cfg, length) for length in cfg.fleet]
    while True:
        fleet = [ships[rng.integers(len(ships))] for ships in options]
        cells = [cell for ship in fleet for cell in ship]
        if len(cells) == len(set(cells)):  # no cell used twice, so no overlap
            return fleet


class BattleshipEnv(gym.Env):
    """One game of Battleship from the shooter's side: sink a hidden random fleet.

    Action: the cell to fire at (0..num_cells-1). Firing at a cell twice is not allowed;
    action_masks() says which cells are still legal.

    Observation: a stack of 3 + (number of ships) grids, each height x width:
        0      cell not fired at yet
        1      miss
        2      hit
        3 + i  the cell of the shot that sank ship i (all zeros while ship i is afloat)

    Reward: reward="shot" gives -1 for every shot, reward="miss" gives -1 for every miss.
    """

    def __init__(self, cfg: GameConfig = GameConfig(), reward: str = "shot"):
        if reward not in ("shot", "miss"):
            raise ValueError("reward must be 'shot' or 'miss'")
        self.cfg = cfg
        self.reward_mode = reward
        self.action_space = spaces.Discrete(cfg.num_cells)
        layers = 3 + len(cfg.fleet)
        self.observation_space = spaces.Box(0, 1, shape=(layers, cfg.height, cfg.width), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # seeds self.np_random
        self.fleet = random_fleet(self.cfg, self.np_random)
        self.ship_at = np.full(self.cfg.num_cells, -1)  # ship index on each cell, -1 for water
        for i, ship in enumerate(self.fleet):
            self.ship_at[list(ship)] = i
        self.hits_left = [len(ship) for ship in self.fleet]  # unhit cells left on each ship
        self.shots = 0
        self.obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.obs[0] = 1  # nothing fired at yet
        return self.obs.copy(), {}

    def step(self, action):
        cell = int(action)
        row, col = divmod(cell, self.cfg.width)
        if self.obs[0, row, col] == 0:
            raise ValueError(f"cell {cell} was already fired at")
        self.obs[0, row, col] = 0
        self.shots += 1

        ship = self.ship_at[cell]
        hit = ship >= 0
        if hit:
            self.obs[2, row, col] = 1
            self.hits_left[ship] -= 1
            if self.hits_left[ship] == 0:  # this shot sank the ship
                self.obs[3 + ship, row, col] = 1
        else:
            self.obs[1, row, col] = 1

        reward = 0.0 if (hit and self.reward_mode == "miss") else -1.0
        terminated = sum(self.hits_left) == 0
        return self.obs.copy(), reward, terminated, False, {"shots": self.shots}

    def action_masks(self) -> np.ndarray:
        """True for every cell that hasn't been fired at yet."""
        return self.obs[0].ravel() == 1
