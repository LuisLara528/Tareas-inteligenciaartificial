"""ε-greedy, Watkins Q-learning, and SARSA on a tabular Q."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from qlearn.env import Env, Transition


def empty_q(env: Env) -> dict[str, dict[str, float]]:
    return {s: {a: 0.0 for a in env.actions} for s in env.nonterminal}


def max_q(Q: dict[str, dict[str, float]], state: str, actions: tuple[str, ...]) -> float:
    if state not in Q:
        return 0.0
    return max(Q[state][a] for a in actions)


def argmax_action(Q: dict[str, dict[str, float]], state: str, actions: tuple[str, ...]) -> str:
    """First action in YAML order wins ties (lecture: L when Q is all zeros)."""
    best = actions[0]
    best_v = Q[state][best]
    for a in actions[1:]:
        if Q[state][a] > best_v:
            best, best_v = a, Q[state][a]
    return best


def epsilon_greedy(
    Q: dict[str, dict[str, float]],
    state: str,
    actions: tuple[str, ...],
    epsilon: float,
    rng: random.Random,
) -> str:
    if state not in Q:
        return actions[0]
    if rng.random() < epsilon:
        return rng.choice(list(actions))
    return argmax_action(Q, state, actions)


def td_target_q(
    env: Env,
    Q: dict[str, dict[str, float]],
    reward: float,
    nxt: str,
    done: bool,
) -> float:
    future = 0.0 if done else max_q(Q, nxt, env.actions)
    return reward + env.gamma * future


def td_target_sarsa(
    env: Env,
    Q: dict[str, dict[str, float]],
    reward: float,
    nxt: str,
    next_action: str | None,
    done: bool,
) -> float:
    if done or next_action is None or nxt not in Q:
        future = 0.0
    else:
        future = Q[nxt][next_action]
    return reward + env.gamma * future


def apply_update(
    Q: dict[str, dict[str, float]],
    state: str,
    action: str,
    target: float,
    alpha: float,
) -> float:
    old = Q[state][action]
    Q[state][action] = old + alpha * (target - old)
    return Q[state][action]


@dataclass
class StepRec:
    state: str
    action: str
    trans: Transition
    target: float
    q_old: float
    q_new: float


@dataclass
class EpisodeRec:
    index: int
    steps: list[StepRec] = field(default_factory=list)
    ret: float = 0.0


def play_scripted(
    env: Env,
    Q: dict[str, dict[str, float]],
    actions: tuple[str, ...],
    ep_index: int,
) -> EpisodeRec:
    rec = EpisodeRec(index=ep_index)
    s = env.start
    for a in actions:
        if s not in Q:
            break
        tr = env.step(s, a)
        target = td_target_q(env, Q, tr.reward, tr.nxt, tr.done)
        old = Q[s][a]
        new = apply_update(Q, s, a, target, env.alpha)
        rec.steps.append(StepRec(s, a, tr, target, old, new))
        rec.ret += tr.reward
        s = tr.nxt
        if tr.done:
            break
    return rec


def train(
    env: Env,
    rng: random.Random,
    episodes: int | None = None,
    epsilon: float | None = None,
    method: str = "q-learning",
) -> tuple[dict[str, dict[str, float]], list[float]]:
    if method not in {"q-learning", "sarsa"}:
        raise ValueError("method must be q-learning or sarsa")
    n = env.episodes if episodes is None else episodes
    eps = env.epsilon if epsilon is None else epsilon
    Q = empty_q(env)
    returns: list[float] = []
    for _ in range(n):
        s = env.start
        G = 0.0
        a = epsilon_greedy(Q, s, env.actions, eps, rng)
        for _step in range(env.max_steps):
            if s not in Q:
                break
            tr = env.step(s, a)
            G += tr.reward
            if method == "q-learning":
                target = td_target_q(env, Q, tr.reward, tr.nxt, tr.done)
                apply_update(Q, s, a, target, env.alpha)
                s = tr.nxt
                if tr.done:
                    break
                a = epsilon_greedy(Q, s, env.actions, eps, rng)
            else:
                a2 = None if tr.done or tr.nxt not in Q else epsilon_greedy(Q, tr.nxt, env.actions, eps, rng)
                target = td_target_sarsa(env, Q, tr.reward, tr.nxt, a2, tr.done)
                apply_update(Q, s, a, target, env.alpha)
                s = tr.nxt
                if tr.done:
                    break
                a = a2 if a2 is not None else env.actions[0]
        returns.append(G)
    return Q, returns


def greedy_policy(env: Env, Q: dict[str, dict[str, float]]) -> dict[str, str]:
    return {s: argmax_action(Q, s, env.actions) for s in env.nonterminal}


def greedy_return(env: Env, Q: dict[str, dict[str, float]]) -> float:
    """Run the greedy policy from start (no exploration)."""
    s = env.start
    G = 0.0
    for _ in range(env.max_steps):
        if s not in Q:
            break
        a = argmax_action(Q, s, env.actions)
        tr = env.step(s, a)
        G += tr.reward
        s = tr.nxt
        if tr.done:
            break
    return G
