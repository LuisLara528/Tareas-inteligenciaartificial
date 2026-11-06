"""Environments loaded from YAML: a graph of (s, a) → (s′, r) or a small grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Transition:
    nxt: str
    reward: float
    done: bool


@dataclass(frozen=True)
class Env:
    name: str
    kind: str
    start: str
    actions: tuple[str, ...]
    nonterminal: tuple[str, ...]
    terminals: tuple[str, ...]
    transitions: dict[tuple[str, str], Transition]
    alpha: float
    gamma: float
    epsilon: float
    episodes: int
    max_steps: int
    seed: int
    walkthrough: tuple[tuple[str, ...], ...]
    rows: int | None = None
    cols: int | None = None
    labels: dict[str, str] | None = None
    source: Path | None = None

    def step(self, state: str, action: str) -> Transition:
        key = (state, action)
        if key not in self.transitions:
            raise ValueError(f"no transition for ({state}, {action})")
        return self.transitions[key]


def load_env(path: str | Path) -> Env:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return parse_env(raw, source=path)


def parse_env(raw: dict[str, Any], source: Path | None = None) -> Env:
    kind = str(raw.get("kind") or "graph")
    actions = tuple(str(a) for a in (raw.get("actions") or ["L", "R"]))
    if kind == "graph":
        start, nonterm, terms, trans, labels, rows, cols = _from_graph(raw, actions)
    elif kind == "grid":
        start, nonterm, terms, trans, labels, rows, cols = _from_grid(raw, actions)
    else:
        raise ValueError(f"unknown kind {kind!r} (graph | grid)")

    walk = tuple(tuple(str(a) for a in ep) for ep in (raw.get("walkthrough") or []))
    return Env(
        name=str(raw.get("name") or "RL environment"),
        kind=kind,
        start=start,
        actions=actions,
        nonterminal=nonterm,
        terminals=terms,
        transitions=trans,
        alpha=float(raw.get("alpha", 0.5)),
        gamma=float(raw.get("gamma", 0.9)),
        epsilon=float(raw.get("epsilon", 0.25)),
        episodes=int(raw.get("episodes", 80)),
        max_steps=int(raw.get("max_steps", 40)),
        seed=int(raw.get("seed", 0)),
        walkthrough=walk,
        rows=rows,
        cols=cols,
        labels=labels,
        source=source,
    )


def _as_state(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return f"{int(value[0])},{int(value[1])}"
    return str(value)


def _from_graph(
    raw: dict[str, Any], actions: tuple[str, ...]
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, str], Transition],
    dict[str, str] | None,
    None,
    None,
]:
    states_raw = raw.get("states") or {}
    if not states_raw:
        raise ValueError("graph: need a states: map")
    start = _as_state(raw.get("start") or next(iter(states_raw)))
    terminals = []
    trans: dict[tuple[str, str], Transition] = {}
    nonterm = []
    for sid, spec in states_raw.items():
        sid = str(sid)
        spec = spec or {}
        if spec.get("terminal"):
            terminals.append(sid)
            continue
        nonterm.append(sid)
        for a in actions:
            if a not in spec:
                raise ValueError(f"state {sid} is missing action {a}")
            edge = spec[a] or {}
            nxt = _as_state(edge.get("to") or sid)
            trans[(sid, a)] = Transition(
                nxt=nxt,
                reward=float(edge.get("r", 0)),
                done=bool(edge.get("terminal", False)),
            )
            if trans[(sid, a)].done and nxt not in terminals:
                terminals.append(nxt)
    # unique, stable order
    seen: list[str] = []
    for s in terminals:
        if s not in seen:
            seen.append(s)
    terminals = seen
    labels = {str(k): str(v) for k, v in (raw.get("labels") or {}).items()} or None
    return start, tuple(nonterm), tuple(terminals), trans, labels, None, None


def _from_grid(
    raw: dict[str, Any], actions: tuple[str, ...]
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    dict[tuple[str, str], Transition],
    dict[str, str],
    int,
    int,
]:
    rows = int(raw.get("rows") or 3)
    cols = int(raw.get("cols") or 3)
    start = _as_state(raw.get("start") or [0, 0])
    goal = _as_state(raw.get("goal") or [0, cols - 1])
    pit = raw.get("pit")
    pit_s = _as_state(pit) if pit is not None else None
    cliffs = {_as_state(c) for c in (raw.get("cliffs") or [])}
    step_r = float(raw.get("step_reward", 0))
    goal_r = float(raw.get("goal_reward", 1))
    pit_r = float(raw.get("pit_reward", -1))
    cliff_r = float(raw.get("cliff_reward", -100))
    delta = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}
    terminals = [goal]
    if pit_s:
        terminals.append(pit_s)
    labels = {start: "S", goal: "G"}
    if pit_s:
        labels[pit_s] = "X"
    for c in cliffs:
        labels[c] = "C"
        if c not in terminals:
            terminals.append(c)

    trans: dict[tuple[str, str], Transition] = {}
    nonterm = []
    for r in range(rows):
        for c in range(cols):
            s = f"{r},{c}"
            if s in terminals:
                continue
            nonterm.append(s)
            for a in actions:
                if a not in delta:
                    raise ValueError(f"grid action {a} must be N, E, S, or W")
                dr, dc = delta[a]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    nr, nc = r, c
                nxt = f"{nr},{nc}"
                if nxt == goal:
                    trans[(s, a)] = Transition(nxt, goal_r, True)
                elif pit_s and nxt == pit_s:
                    trans[(s, a)] = Transition(nxt, pit_r, True)
                elif nxt in cliffs:
                    trans[(s, a)] = Transition(nxt, cliff_r, True)
                else:
                    trans[(s, a)] = Transition(nxt, step_r, False)
    return start, tuple(nonterm), tuple(terminals), trans, labels, rows, cols
