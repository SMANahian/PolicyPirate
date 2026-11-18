"""Train a PPO agent on Battleship.

Example:
    python -m policypirate.train --net conv --reward shot --steps 2000000
"""

import argparse
import os

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from policypirate.game import BattleshipEnv
from policypirate.networks import CnnDense, ConvPolicy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent on Battleship.")
    parser.add_argument("--net", choices=["cnn_dense", "conv"], default="conv")
    parser.add_argument("--reward", choices=["shot", "miss"], default="shot")
    parser.add_argument("--steps", type=int, default=2_000_000, help="total shots to train for")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--gamma", type=float, default=1.0, help="1.0 = minimize total shots exactly")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    name = f"{args.net}_{args.reward}_seed{args.seed}"

    # 16 games in parallel. Board seeds start at 1,000,000, so training never starts on an
    # evaluation board (0-999), and each --seed gets its own boards.
    env = make_vec_env(
        BattleshipEnv, n_envs=16, seed=1_000_000 + 100 * args.seed, env_kwargs=dict(reward=args.reward)
    )
    if args.net == "cnn_dense":
        policy, policy_kwargs = "MlpPolicy", dict(features_extractor_class=CnnDense, net_arch=[])
    else:
        policy, policy_kwargs = ConvPolicy, {}

    model = MaskablePPO(
        policy,
        env,
        policy_kwargs=policy_kwargs,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
        tensorboard_log="runs",
    )
    model.learn(total_timesteps=args.steps, tb_log_name=name)
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{name}")
    print(f"saved models/{name}.zip")
