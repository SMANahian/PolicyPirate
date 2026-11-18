"""Baseline bots.

Each bot is a function (obs, mask, rng) -> cell to fire at. It only uses what the agent sees:
the observation from BattleshipEnv (layer 0 not fired yet, 1 miss, 2 hit, 3+ sunk marks)
and the mask of legal cells.
"""

from functools import lru_cache

import numpy as np

from policypirate.game import GameConfig, ship_placements


def random_bot(obs, mask, rng):
    """Fire at a random legal cell."""
    return int(rng.choice(np.flatnonzero(mask)))


def next_to_hits(obs, mask):
    """Legal cells directly above, below, left or right of a hit."""
    hits = obs[2] == 1
    near = np.zeros_like(hits)
    near[1:, :] |= hits[:-1, :]  # cell below a hit
    near[:-1, :] |= hits[1:, :]  # cell above a hit
    near[:, 1:] |= hits[:, :-1]  # cell right of a hit
    near[:, :-1] |= hits[:, 1:]  # cell left of a hit
    return near.ravel() & mask


def hunt_target(obs, mask, rng, parity=False):
    """Target mode: fire next to a hit if there is such a cell. Hunt mode: fire at random.

    With parity=True, hunting only fires on a checkerboard pattern: every ship is at least
    2 long, so it always covers at least one checkerboard cell.
    """
    targets = next_to_hits(obs, mask)
    if targets.any():
        return int(rng.choice(np.flatnonzero(targets)))
    if parity:
        height, width = obs.shape[1:]
        checkerboard = (np.add.outer(np.arange(height), np.arange(width)) % 2 == 0).ravel()
        if (mask & checkerboard).any():
            return int(rng.choice(np.flatnonzero(mask & checkerboard)))
    return random_bot(obs, mask, rng)


def hunt_target_parity(obs, mask, rng):
    return hunt_target(obs, mask, rng, parity=True)


@lru_cache
def placement_grid(cfg, length):
    """ship_placements as a (placements, cells) True/False array: row p marks placement p's cells."""
    places = ship_placements(cfg, length)
    grid = np.zeros((len(places), cfg.num_cells), dtype=bool)
    for p, cells in enumerate(places):
        grid[p, list(cells)] = True
    return grid


TARGET_WEIGHT = 100  # the "heavy" extra weight for each unexplained hit a placement covers


def sunk_ship_cells(obs, cfg):
    """Cells that surely belong to sunk ships.

    A sunk ship lies through the cell of its final shot, on hits only, and not on cells already
    known to belong to another sunk ship. The cells shared by all such positions surely belong
    to it. Each pass can settle a ship that helps settle another, so we repeat once per ship.
    """
    hit = obs[2].ravel() == 1
    owned = [np.zeros(cfg.num_cells, dtype=bool) for _ in cfg.fleet]  # known cells of each ship
    for _ in cfg.fleet:
        for i, length in enumerate(cfg.fleet):
            final_shot = obs[3 + i].ravel() == 1
            if final_shot.any():
                others = np.any([owned[j] for j in range(len(cfg.fleet)) if j != i], axis=0)
                places = placement_grid(cfg, length)
                fits = (places & final_shot).any(axis=1) & ~(places & (~hit | others)).any(axis=1)
                owned[i] = places[fits].all(axis=0)
    return np.any(owned, axis=0)


def probability_density(obs, mask, rng, cfg=GameConfig()):
    """Fire at the cell the ships still afloat could cover in the most ways (DataGenetics, 2011).

    Misses and sunk ships are obstacles. A placement counts 1, plus TARGET_WEIGHT for every
    unexplained hit it passes through, so the bot finishes off hit ships before hunting again.
    """
    miss = obs[1].ravel() == 1
    sunk = sunk_ship_cells(obs, cfg)
    open_hits = (obs[2].ravel() == 1) & ~sunk

    density = np.zeros(cfg.num_cells)
    for i, length in enumerate(cfg.fleet):
        if obs[3 + i].any():  # this ship is sunk
            continue
        places = placement_grid(cfg, length)
        places = places[~(places & (miss | sunk)).any(axis=1)]  # can't cover a miss or a sunk ship
        weight = 1 + TARGET_WEIGHT * (places & open_hits).sum(axis=1)
        density += weight @ places
    density[~mask] = -1
    return int(rng.choice(np.flatnonzero(density == density.max())))
