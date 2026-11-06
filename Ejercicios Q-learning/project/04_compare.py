#!/usr/bin/env python3
"""Program 4: exploration, Q-learning vs SARSA, and a learned grid policy."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qlearn.agent import greedy_return, train  # noqa: E402
from qlearn.cli import ROOT as PROJ  # noqa: E402
from qlearn.env import load_env  # noqa: E402
from qlearn.format import format_grid_policy, format_q  # noqa: E402

ENVS = PROJ / "envs"


def explore_demo() -> None:
    env = load_env(ENVS / "corridor.yaml")
    print("1. Explore or get stuck")
    print("-" * 72)
    print(env.name)
    print("Q starts at 0. Ties pick L. From A, L stays in A.")
    print()
    print(f"{'ε':>8}  {'mean return (last 20)':>22}  {'greedy return':>14}  Q(A,R)   Q(B,R)")
    for eps, seed in ((0.0, env.seed), (0.25, env.seed)):
        Q, rets = train(env, random.Random(seed), epsilon=eps, method="q-learning")
        tail = rets[-20:]
        mean = sum(tail) / len(tail)
        print(
            f"{eps:8.2f}  {mean:22.3f}  {greedy_return(env, Q):14.3f}  "
            f"{Q['A']['R']:6.3f}  {Q['B']['R']:6.3f}"
        )
    print()
    print("ε = 0 never sees G, so both Q(·, R) stay 0. ε-greedy finds R and the reward backs up.")


def sarsa_demo() -> None:
    env = load_env(ENVS / "cliff.yaml")
    print("2. Off-policy vs on-policy (cliff)")
    print("-" * 72)
    print(env.name)
    print("Step −1, cliff −100, goal 0. Q-learning uses max Q(s′, ·); SARSA uses the action you took.")
    print()
    for method in ("q-learning", "sarsa"):
        Q, rets = train(env, random.Random(env.seed), method=method)
        tail = rets[-50:]
        print(f"{method}")
        print(f"  mean return (last 50) = {sum(tail) / len(tail):.1f}    greedy return = {greedy_return(env, Q):g}")
        print(format_grid_policy(env, Q))
        print()
    print("Q-learning walks the row next to the cliff (greedy return −5). SARSA stays higher (−7): it assumes ε-greedy will keep exploring.")


def grid_demo() -> None:
    env = load_env(ENVS / "grid.yaml")
    print("3. Greedy arrows after Q-learning")
    print("-" * 72)
    print(env.name)
    Q, _ = train(env, random.Random(env.seed), method="q-learning")
    print(format_q(env, Q))
    print()
    print(format_grid_policy(env, Q))
    print()
    print(f"greedy return from S: {greedy_return(env, Q):g}   (pit is X, goal is G)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exploration, SARSA, and a grid policy.")
    parser.add_argument("--only", choices=("explore", "sarsa", "grid", "all"), default="all")
    args = parser.parse_args()
    demos = {"explore": explore_demo, "sarsa": sarsa_demo, "grid": grid_demo}
    names = list(demos) if args.only == "all" else [args.only]
    for i, name in enumerate(names):
        if i:
            print()
            print()
        demos[name]()


if __name__ == "__main__":
    main()
