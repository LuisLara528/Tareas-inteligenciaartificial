#!/usr/bin/env python3
"""Program 1: print the environment (states, actions, transitions, knobs)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from qlearn.cli import build_parser, load  # noqa: E402
from qlearn.format import format_env  # noqa: E402


def main() -> None:
    parser = build_parser("Show a YAML RL environment.")
    args = parser.parse_args()
    env = load(args)
    print(format_env(env))
    print()
    print("The agent does not see a label y. It only sees s, chooses a, and gets r and s′.")
    print("Edit the YAML file, then rerun this program.")


if __name__ == "__main__":
    main()
