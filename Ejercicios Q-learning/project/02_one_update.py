#!/usr/bin/env python3
"""Program 2: one Watkins update at a time (lecture walkthrough by default)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qlearn.agent import empty_q, play_scripted  # noqa: E402
from qlearn.cli import DEFAULT_CORRIDOR, build_parser, load  # noqa: E402
from qlearn.format import format_q, format_step  # noqa: E402


def main() -> None:
    parser = build_parser("Replay scripted episodes and print every TD update.", default=DEFAULT_CORRIDOR)
    args = parser.parse_args()
    env = load(args)
    if not env.walkthrough:
        raise SystemExit(
            f"{env.source}: add walkthrough: lists of actions, e.g. [[R, R], [R, R]]"
        )

    print(env.name)
    print()
    print("Q(s, a) ← Q(s, a) + α [ r + γ max_a′ Q(s′, a′) − Q(s, a) ]")
    print(f"α = {env.alpha}    γ = {env.gamma}    Q starts at 0")
    print("If s′ is terminal, max Q(s′) = 0.")
    print()

    Q = empty_q(env)
    for i, actions in enumerate(env.walkthrough, 1):
        rec = play_scripted(env, Q, actions, i)
        print(format_step(rec))
        print()

    print("Q after the walkthrough")
    print(format_q(env, Q))


if __name__ == "__main__":
    main()
