"""CLI helpers."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from qlearn.env import Env, load_env

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORRIDOR = ROOT / "envs" / "corridor.yaml"
DEFAULT_GRID = ROOT / "envs" / "grid.yaml"


def build_parser(description: str, default: Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--env", type=Path, default=default or DEFAULT_CORRIDOR, help="YAML environment")
    parser.add_argument("--seed", type=int, default=None, help="Override YAML seed")
    parser.add_argument("--episodes", type=int, default=None, help="Override YAML episodes")
    parser.add_argument("--epsilon", type=float, default=None, help="Override YAML ε")
    parser.add_argument("--alpha", type=float, default=None, help="Override YAML α")
    parser.add_argument("--gamma", type=float, default=None, help="Override YAML γ")
    return parser


def load(args: argparse.Namespace) -> Env:
    return load_env(args.env)


def make_rng(env: Env, seed_override: int | None) -> random.Random:
    seed = seed_override if seed_override is not None else env.seed
    return random.Random(seed)
