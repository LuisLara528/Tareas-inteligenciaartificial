"""Pretty-print environments, Q-tables, and greedy policies."""

from __future__ import annotations

from qlearn.agent import EpisodeRec, greedy_policy
from qlearn.env import Env


def format_env(env: Env) -> str:
    lines = [
        env.name,
        "",
        f"kind = {env.kind}    start = {env.start}",
        f"actions = {', '.join(env.actions)}",
        f"α = {env.alpha}    γ = {env.gamma}    ε = {env.epsilon}",
        f"episodes = {env.episodes}    max_steps = {env.max_steps}    seed = {env.seed}",
        "",
        f"{'s':<8}  {'a':<4}  {'s′':<8}  {'r':>6}  done",
        "-" * 42,
    ]
    for s in env.nonterminal:
        for a in env.actions:
            tr = env.transitions[(s, a)]
            done = "yes" if tr.done else "no"
            lines.append(f"{s:<8}  {a:<4}  {tr.nxt:<8}  {tr.reward:6.2f}  {done}")
    if env.terminals:
        lines.append("")
        lines.append("terminals: " + ", ".join(env.terminals))
    return "\n".join(lines)


def format_q(env: Env, Q: dict[str, dict[str, float]]) -> str:
    act_w = max(4, *(len(a) for a in env.actions))
    s_w = max(6, *(len(s) for s in env.nonterminal))
    header = f"{'s':<{s_w}}" + "".join(f"  {a:>{act_w}}" for a in env.actions) + "  greedy"
    lines = [header, "-" * len(header)]
    policy = greedy_policy(env, Q)
    for s in env.nonterminal:
        row = f"{s:<{s_w}}"
        for a in env.actions:
            row += f"  {Q[s][a]:>{act_w}.3f}"
        row += f"  {policy[s]}"
        lines.append(row)
    return "\n".join(lines)


def format_step(rec: EpisodeRec) -> str:
    lines = [f"Episode {rec.index}    return = {rec.ret:g}"]
    lines.append("-" * 72)
    for i, st in enumerate(rec.steps, 1):
        tr = st.trans
        lines.append(
            f"{i}.  s={st.state}  a={st.action}  r={tr.reward:g}  s′={tr.nxt}"
            + ("  terminal" if tr.done else "")
        )
        lines.append(
            f"    TD objective = {st.target:.3f}    "
            f"Q({st.state},{st.action})  {st.q_old:.3f} → {st.q_new:.3f}"
        )
    return "\n".join(lines)


def sparkline(values: list[float], width: int = 40) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi > lo else 1.0
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    chars = []
    for v in values[::step]:
        idx = int(round((v - lo) / span * (len(blocks) - 1)))
        idx = max(0, min(len(blocks) - 1, idx))
        chars.append(blocks[idx])
    return "".join(chars)


def format_returns(returns: list[float], tail: int = 20) -> str:
    last = returns[-tail:] if returns else []
    mean = sum(last) / len(last) if last else 0.0
    lines = [
        f"{len(returns)} episodes    mean return (last {len(last)}) = {mean:.3f}",
        f"sparkline: {sparkline(returns)}",
    ]
    return "\n".join(lines)


_ARROWS = {"N": "↑", "E": "→", "S": "↓", "W": "←", "L": "←", "R": "→"}


def format_grid_policy(env: Env, Q: dict[str, dict[str, float]]) -> str:
    if env.rows is None or env.cols is None:
        return format_q(env, Q)
    policy = greedy_policy(env, Q)
    labels = env.labels or {}
    lines = []
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            s = f"{r},{c}"
            if s in labels and s not in env.nonterminal:
                cells.append(f"{labels[s]:^3}")
            elif s in policy:
                mark = _ARROWS.get(policy[s], policy[s][:1])
                tag = labels.get(s, " ")
                cells.append(f"{tag}{mark} ")
            else:
                cells.append(" · ")
        lines.append(" ".join(cells))
    return "\n".join(lines)
