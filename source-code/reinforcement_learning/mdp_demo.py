import mdptoolbox.example
import numpy as np

print("=" * 55)
print("Markov Decision Process Demo")
print("=" * 55)

print("\n--- Example 1: Custom 3x3 Grid World ---")
n_states = 9
n_actions = 4

P = np.zeros((n_actions, n_states, n_states))
grid = [(r, c) for r in range(3) for c in range(3)]
for s, (r, c) in enumerate(grid):
    for a, (dr, dc) in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            ns = nr * 3 + nc
        else:
            ns = s
        P[a, s, ns] = 1.0

R = np.zeros((n_states, n_actions))
R[8, :] = 10.0
R[5, :] = -5.0

vi = mdptoolbox.mdp.ValueIteration(P, R, 0.9)
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

print("\n--- Policy Iteration on same grid ---")
pi = mdptoolbox.mdp.PolicyIteration(P, R, 0.9)
pi.run()
print(f"Policy: {tuple(pi.policy)}")
print(f"Iterations: {pi.iter}")

print("\n--- Example 2: Forest Management (built-in) ---")
P2, R2 = mdptoolbox.example.forest(S=5, r1=4, r2=2, p=0.1)
print("States: 5 (forest age classes 0-4)")
print("Action 0 = Wait, Action 1 = Cut")
print("p(fire) = 0.1 each year")
print(f"Reward shape: {R2.shape} (states × actions)")

vi2 = mdptoolbox.mdp.ValueIteration(P2, R2, 0.9)
vi2.run()
print(f"Optimal policy: {tuple(vi2.policy)}")
for s, a in enumerate(vi2.policy):
    print(f"  Forest age {s}: {'Wait' if a == 0 else 'Cut'}")
print(f"Value function: {np.round(vi2.V, 2)}")
print(f"Iterations: {vi2.iter}")
