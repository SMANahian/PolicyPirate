"""Play bots on the same fixed boards and compare how many shots they need."""

import numpy as np

from policypirate.bots import hunt_target, hunt_target_parity, probability_density, random_bot
from policypirate.game import BattleshipEnv


def play(bot, seed):
    """Play one game with `bot` on the board given by `seed`; return the number of shots."""
    env = BattleshipEnv()
    obs, _ = env.reset(seed=seed)  # the seed fixes where the ships are
    rng = np.random.default_rng([1, seed])  # the bot's own random stream, separate from the board's
    done = False
    while not done:
        obs, _, done, _, info = env.step(bot(obs, env.action_masks(), rng))
    return info["shots"]


def evaluate(bot, num_boards=1000):
    """Shots needed on boards 0..num_boards-1: mean, 95% confidence interval, median, worst."""
    shots = np.array([play(bot, seed) for seed in range(num_boards)])
    ci = 1.96 * shots.std(ddof=1) / np.sqrt(num_boards)
    return {"mean": shots.mean(), "ci": ci, "median": np.median(shots), "worst": shots.max()}


if __name__ == "__main__":
    bots = {
        "random": random_bot,
        "hunt/target": hunt_target,
        "hunt/target + parity": hunt_target_parity,
        "probability density": probability_density,
    }
    print(f"{'bot':<22}{'mean':>12}{'median':>8}{'worst':>7}")
    for name, bot in bots.items():
        r = evaluate(bot)
        print(f"{name:<22}{r['mean']:>6.1f} ± {r['ci']:.1f}{r['median']:>8.0f}{r['worst']:>7}")
