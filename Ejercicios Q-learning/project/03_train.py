#!/usr/bin/env python3
"""Program 3: train Q-learning (or SARSA) for several episodes."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qlearn.agent import greedy_return, train  # noqa: E402
from qlearn.cli import build_parser, load, make_rng  # noqa: E402
from qlearn.format import format_env, format_grid_policy, format_q, format_returns  # noqa: E402


def main() -> None:
    parser = build_parser("Train tabular Q-learning / SARSA.")
    parser.add_argument(
        "--method",
        choices=("q-learning", "sarsa"),
        default="q-learning",
        help="Update rule (default: q-learning, off-policy)",
    )
    args = parser.parse_args()
    env = load(args)
    env = replace(
        env,
        seed=args.seed if args.seed is not None else env.seed,
        episodes=args.episodes if args.episodes is not None else env.episodes,
        epsilon=args.epsilon if args.epsilon is not None else env.epsilon,
        alpha=args.alpha if args.alpha is not None else env.alpha,
        gamma=args.gamma if args.gamma is not None else env.gamma,
    )
    rng = make_rng(env, None)

    print(format_env(env))
    print()
    print(f"method = {args.method}    behaviour = ε-greedy")
    print()

    Q, returns = train(env, rng, method=args.method)
    print("Q")
    print("-" * 72)
    print(format_q(env, Q))
    print()
    if env.kind == "grid":
        print("Greedy policy (arrows)")
        print("-" * 72)
        print(format_grid_policy(env, Q))
        print()
    print(format_returns(returns))
    print(f"greedy return from start (ε = 0): {greedy_return(env, Q):g}")


if __name__ == "__main__":
    main()
