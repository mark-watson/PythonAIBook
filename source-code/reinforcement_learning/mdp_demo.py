# Markov Decision Process (MDP) Demo
# Demonstrates exact solution methods — Value Iteration and Policy Iteration —
# on a hand-built 3×3 grid world and the built-in Forest Management example
# from pymdptoolbox.
#
# References:
#   MDP:              https://en.wikipedia.org/wiki/Markov_decision_process
#   Value Iteration:  https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
#   Policy Iteration: https://en.wikipedia.org/wiki/Markov_decision_process#Policy_iteration
#   Bellman equation: https://en.wikipedia.org/wiki/Bellman_equation
#   pymdptoolbox:     https://pymdptoolbox.readthedocs.io/

import mdptoolbox.example   # Pre-built MDP models (e.g., forest management)
import numpy as np

print("=" * 55)
print("Markov Decision Process Demo")
print("=" * 55)

# =============================================================================
# Example 1: Custom 3×3 Grid World
#
# A 3×3 grid with 9 discrete states (indexed 0..8, row-major). The agent can
# move in 4 directions (↑→↓←). Bumping into a wall leaves the agent in the
# same cell (self-loop). The goal cell (8, lower-right) gives +10 reward; the
# trap cell (5, center-right) gives −5. All other cells give 0.
#
# Transition matrix P[a, s, ns] = probability of moving from state s to
# next state ns given action a. The environment is deterministic, so each
# (s, a) pair maps to exactly one ns with probability 1.0.
# =============================================================================

print("\n--- Example 1: Custom 3x3 Grid World ---")
n_states = 9       # 3 rows × 3 cols
n_actions = 4      # ↑ (0), → (1), ↓ (2), ← (3)

# Build deterministic transition matrix:
# P shape = (n_actions, n_states, n_states)
P = np.zeros((n_actions, n_states, n_states))
grid = [(r, c) for r in range(3) for c in range(3)]
for s, (r, c) in enumerate(grid):
    # Action deltas: up=(-1,0), right=(0,1), down=(1,0), left=(0,-1)
    for a, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            ns = nr * 3 + nc          # valid move — transition to new cell
        else:
            ns = s                     # wall bump — stay in same cell
        P[a, s, ns] = 1.0             # deterministic: probability = 1.0

# Reward matrix R[s, a] = immediate reward for taking action a in state s
R = np.zeros((n_states, n_actions))
R[8, :] = 10.0    # goal cell (bottom-right) — all actions yield +10
R[5, :] = -5.0    # trap cell (center-right) — all actions yield −5

# --- Value Iteration ---
# Iteratively updates the value function using the Bellman optimality equation:
#   V(s) ← max_a [ R(s,a) + γ Σ_{s'} P(s'|s,a) V(s') ]
# until convergence (within tolerance).
# See: https://en.wikipedia.org/wiki/Value_iteration
vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)  # discount factor γ = 0.9
vi.run()
actions = ["↑", "→", "↓", "←"]
print("Optimal policy:")
for r in range(3):
    row = ""
    for c in range(3):
        s = r * 3 + c
        row += f"  {actions[vi.policy[s]]}  "
    print(row)
print(f"\nValue function: {np.round(vi.V, 2)}")
print(f"Iterations to converge: {vi.iter}")

# --- Policy Iteration ---
# Alternates between *policy evaluation* (compute V for the current policy by
# solving a linear system) and *policy improvement* (make the policy greedy
# w.r.t. the current V). Converges in fewer iterations than Value Iteration
# but each iteration is more expensive.
# See: https://en.wikipedia.org/wiki/Policy_iteration
print("\n--- Policy Iteration on same grid ---")
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
pi.run()
print(f"Policy: {tuple(pi.policy)}")
print(f"Iterations: {pi.iter}")

# =============================================================================
# Example 2: Forest Management (built-in model from pymdptoolbox)
#
# A classic MDP problem: a forest manager decides each year whether to
# Wait (let the forest grow another year) or Cut (harvest and replant).
# Fire occurs with probability p = 0.1, resetting the forest to age 0.
# States represent forest age classes (0 = youngest, 4 = oldest).
#
# Parameters:
#   S = 5      number of states (age classes)
#   r1 = 4     reward when cutting age ≥ 1 (older trees yield more)
#   r2 = 2     reward when cutting age 0 (young trees yield less)
#   p  = 0.1   probability of fire each year
#
# See: https://pymdptoolbox.readthedocs.io/en/latest/api/example.html
# =============================================================================

print("\n--- Example 2: Forest Management (built-in) ---")
P2, R2 = mdptoolbox.example.forest(S=5, r1=4, r2=2, p=0.1)
print("States: 5 (forest age classes 0-4)")
print("Action 0 = Wait, Action 1 = Cut")
print("p(fire) = 0.1 each year")
print(f"Reward shape: {R2.shape} (states × actions)")

# Solve with Value Iteration
vi2 = mdptoolbox.mdp.ValueIteration(P2, R2, 0.9)
vi2.run()
print(f"Optimal policy: {tuple(vi2.policy)}")
for s, a in enumerate(vi2.policy):
    print(f"  Forest age {s}: {'Wait' if a == 0 else 'Cut'}")
print(f"Value function: {np.round(vi2.V, 2)}")
print(f"Iterations: {vi2.iter}")