"""
Ejercicio 1: Q-learning en Cliff Walking (Sutton & Barto)

"""

from __future__ import annotations
import random
from dataclasses import dataclass
import matplotlib.pyplot as plt


# 1. ENTORNO CLIFF WALKING (4x12)
@dataclass(frozen=True)
class Transition:
    nxt: str
    reward: float
    done: bool


class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = "3,0"
        self.goal = "3,11"
        self.cliffs = {f"3,{c}" for c in range(1, 11)}
        self.actions = ("N", "E", "S", "W")
        self.alpha = 0.5
        self.gamma = 1.0
        self.epsilon = 0.1
        self.episodes = 500
        self.max_steps = 200
        self.delta = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}

        # Estados no terminales: toda la cuadricula excepto la meta (G)
        self.nonterminal = tuple(
            f"{r},{c}"
            for r in range(self.rows)
            for c in range(self.cols)
            if f"{r},{c}" != self.goal and f"{r},{c}" not in self.cliffs
        )

    def step(self, state: str, action: str) -> Transition:
        r, c = map(int, state.split(","))
        dr, dc = self.delta[action]
        nr, nc = r + dr, c + dc

        # Si choca contra el borde del mapa, permanece en la misma casilla
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c

        nxt = f"{nr},{nc}"

        # Reglas de Sutton & Barto (Ejemplo Cliff Walking):
        # - Caer al acantilado: penalizacion de -100, regresa a S (3,0), el episodio continua (done=False)
        if nxt in self.cliffs:
            return Transition(nxt=self.start, reward=-100.0, done=False)
        # - Llegar a la meta (G): recompensa de -1, el episodio termina (done=True)
        elif nxt == self.goal:
            return Transition(nxt=self.goal, reward=-1.0, done=True)
        # - Paso ordinario: recompensa de -1, el episodio continua (done=False)
        else:
            return Transition(nxt=nxt, reward=-1.0, done=False)


# 2. FUNCIONES DE AGENTE 

def empty_q(env: CliffWalkingEnv) -> dict[str, dict[str, float]]:
    return {s: {a: 0.0 for a in env.actions} for s in env.nonterminal}


def max_q(Q: dict[str, dict[str, float]], state: str, actions: tuple[str, ...]) -> float:
    if state not in Q:
        return 0.0
    return max(Q[state][a] for a in actions)


def argmax_action(Q: dict[str, dict[str, float]], state: str, actions: tuple[str, ...]) -> str:
    # Desempate en orden fijo: N, E, S, W
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
    env: CliffWalkingEnv,
    Q: dict[str, dict[str, float]],
    reward: float,
    nxt: str,
    done: bool,
) -> float:
    # Si s' es la meta (terminal), el valor futuro es 0.0
    # Si cayo al acantilado, s' es S (3,0) que si esta en Q
    future = 0.0 if done else max_q(Q, nxt, env.actions)
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


# 3. ENTRENAMIENTO DE Q-LEARNING
def train(env: CliffWalkingEnv, rng: random.Random) -> tuple[dict[str, dict[str, float]], list[float]]:
    Q = empty_q(env)
    returns: list[float] = []

    for ep in range(env.episodes):
        s = env.start
        G = 0.0
        for _step in range(env.max_steps):
            if s not in Q:
                break
            a = epsilon_greedy(Q, s, env.actions, env.epsilon, rng)
            tr = env.step(s, a)
            G += tr.reward

            target = td_target_q(env, Q, tr.reward, tr.nxt, tr.done)
            apply_update(Q, s, a, target, env.alpha)

            s = tr.nxt
            if tr.done:
                break
        returns.append(G)

    return Q, returns


# 4. EVALUACIÓN POLÍTICA GREEDY (epsilon = 0)
def evaluate_greedy(env: CliffWalkingEnv, Q: dict[str, dict[str, float]]) -> tuple[float, list[str], list[str]]:
    s = env.start
    G = 0.0
    path = [s]
    actions_taken = []

    for _ in range(env.max_steps):
        if s not in Q:
            break
        a = argmax_action(Q, s, env.actions)
        actions_taken.append(a)
        tr = env.step(s, a)
        G += tr.reward
        path.append(tr.nxt)
        s = tr.nxt
        if tr.done:
            break

    return G, actions_taken, path


def print_grid_policy(env: CliffWalkingEnv, Q: dict[str, dict[str, float]]) -> None:
    arrows = {"N": "↑", "E": "→", "S": "↓", "W": "←"}
    print("\nPolitica greedy aprendida (cuadricula 4x12):")
    print("-" * 50)
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            s = f"{r},{c}"
            if s == env.start:
                row.append(" S ")
            elif s == env.goal:
                row.append(" G ")
            elif s in env.cliffs:
                row.append(" C ")
            elif s in Q:
                best_a = argmax_action(Q, s, env.actions)
                row.append(f" {arrows[best_a]} ")
            else:
                row.append(" . ")
        print("".join(row))
    print("-" * 50)


# 5. GRAFICAR CURVA DE APRENDIZAJE
def plot_returns(returns: list[float], window: int = 10, filename: str = "curva_aprendizaje.png") -> None:
    smoothed = []
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        smoothed.append(sum(returns[start : i + 1]) / (i - start + 1))

    plt.figure(figsize=(9, 4.5))
    plt.plot(returns, color="lightgray", label="Recompensa por episodio")
    plt.plot(smoothed, color="blue", linewidth=1.8, label=f"Media movil ({window} episodios)")
    plt.axhline(y=-13, color="red", linestyle="--", label="Retorno optimo greedy (-13)")
    plt.ylim(-150, 0)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa acumulada")
    plt.title("Q-learning en Cliff Walking (4x12)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\nGrafica guardada exitosamente en: {filename}")



# MAIN


if __name__ == "__main__":
    SEED = 42
    rng = random.Random(SEED)
    env = CliffWalkingEnv()

    print(f"Entrenando Q-learning en Cliff Walking 4x12 (Semilla={SEED}, Episodios={env.episodes})...")
    Q, returns = train(env, rng)

    ret, actions, path = evaluate_greedy(env, Q)
    print("\n")
    print("RESULTADOS DE EVALUACION GREEDY (epsilon = 0):")
    print("\n")
    print(f"Pasos totales : {len(actions)} (Esperado: 13)")
    print(f"Retorno greedy: {ret:.1f} (Esperado: -13.0)")
    print(f"Secuencia     : {' -> '.join(actions)}")
    print(f"Camino        : {' -> '.join(path)}")

    print_grid_policy(env, Q)
    plot_returns(returns, window=10)