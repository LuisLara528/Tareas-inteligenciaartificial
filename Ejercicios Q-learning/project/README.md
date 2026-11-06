# Q-learning

Watkins **Q-learning** you can read line by line (no NumPy, no Gym): ε-greedy
behaviour, one TD update per step. The default world is the lecture example:
a **3-cell corridor** A → B → G.

## Setup

```bash
cd "Q-learning/project"
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows (PowerShell):

```powershell
cd "Q-learning/project"
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Later sessions: activate the venv again, then run the programs.
Deactivate with `deactivate`.

## Programs

| File | Role |
|---|---|
| `01_show_env.py` | States, actions, transitions, α / γ / ε |
| `02_one_update.py` | Every TD update on a scripted path |
| `03_train.py` | ε-greedy episodes; Q-table and greedy policy |
| `04_compare.py` | Exploration, Q-learning vs SARSA, grid arrows |

```bash
python 01_show_env.py
python 02_one_update.py
python 03_train.py
python 04_compare.py
python 01_show_env.py --env envs/grid.yaml
python 03_train.py --env envs/grid.yaml
python 03_train.py --method sarsa --env envs/cliff.yaml
python 04_compare.py --only explore
python 04_compare.py --only sarsa
python 04_compare.py --only grid
```

## Edit the world

Open a file in `envs/` (or copy one).

**`graph`** — name every `(s, a)`:

```yaml
name: Lecture 3-cell corridor
kind: graph
start: A
actions: [L, R]
alpha: 0.5
gamma: 0.9
epsilon: 0.25
episodes: 80
seed: 2

walkthrough:
  - [R, R]
  - [R, R]

states:
  A:
    L: {to: A, r: 0.0, terminal: false}
    R: {to: B, r: 0.0, terminal: false}
  B:
    L: {to: A, r: 0.0, terminal: false}
    R: {to: G, r: 1.0, terminal: true}
  G:
    terminal: true
```

**`grid`** — a rectangle; bumping a wall leaves you in place:

```yaml
kind: grid
rows: 3
cols: 3
start: [0, 0]
goal: [0, 2]
pit: [2, 2]              # optional
cliffs: [[2, 1], [2, 2]] # optional; episode ends
actions: [N, E, S, W]
step_reward: 0.0
goal_reward: 1.0
pit_reward: -1.0
cliff_reward: -100.0
```

`walkthrough` is the scripted action lists for program 02. Ties in arg max
follow **YAML action order** (so `[L, R]` with Q = 0 always picks L).

## What to expect

**`envs/corridor.yaml`** (program 02) replays the lecture:

- α = 0.5, γ = 0.9, Q starts at 0
- Ep 1: Q(A,R) stays 0; Q(B,R) ← **0.50**
- Ep 2: Q(A,R) ← **0.225**; Q(B,R) ← **0.75**
- After that, greedy is already R in A and in B

**`03_train.py`** on the corridor (80 episodes, ε = 0.25) — Q(B,R) → **1**,
Q(A,R) → **0.9**; greedy return from A is 1.

**`04_compare.py`**

- Explore: ε = 0 never leaves A (mean return 0). ε = 0.25 finds G.
- Cliff: Q-learning’s greedy path walks the row next to the cliff (return
  **−5**). SARSA stays on the top row (**−7**) because its target uses the
  action it actually took, including the random ones.
- Grid: after 400 episodes, arrows point toward G and away from the pit X.
