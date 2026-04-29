import gymnasium as gym
import numpy as np

np.random.seed(42)


def q_learning(env, episodes=10000, alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    success_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (
                reward + gamma * best_next - Q[state, action]
            )

            state = next_state

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % 1000 == 0:
            success_rate = evaluate(env, Q, episodes=100)
            success_history.append((episode + 1, success_rate))
            print(f"  Episode {episode + 1:>5}: success rate = {success_rate:.2f}")

    return Q, success_history


def evaluate(env, Q, episodes=100):
    successes = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward == 1.0:
                successes += 1
    return successes / episodes


def print_policy(Q, env_name):
    n_states = Q.shape[0]
    size = int(np.sqrt(n_states))
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    print(f"\nLearned policy for {env_name}:")
    for row in range(size):
        row_str = ""
        for col in range(size):
            s = row * size + col
            row_str += arrows[np.argmax(Q[s])]
        print(f"  {row_str}")
    print("  (←=left ↓=down →=right ↑=up)")


print("=" * 55)
print("Q-Learning on FrozenLake (slippery)")
print("=" * 55)

env = gym.make("FrozenLake-v1", is_slippery=True)
print(f"States: {env.observation_space.n}")
print(f"Actions: {env.action_space.n}")
print(f"Map size: 4x4")

print("\nTraining:")
Q, history = q_learning(env)

print_policy(Q, "FrozenLake-v1 (slippery)")

print("\nFinal evaluation (1000 episodes):")
final_rate = evaluate(env, Q, episodes=1000)
print(f"  Success rate: {final_rate:.2f}")

print("\n" + "=" * 55)
print("Q-Learning on FrozenLake (deterministic, no slipping)")
print("=" * 55)

env2 = gym.make("FrozenLake-v1", is_slippery=False)
print(f"States: {env2.observation_space.n}")
print(f"Actions: {env2.action_space.n}")

print("\nTraining:")
Q2, _ = q_learning(env2, episodes=2000)

print_policy(Q2, "FrozenLake-v1 (deterministic)")

print("\nFinal evaluation (1000 episodes):")
final_rate2 = evaluate(env2, Q2, episodes=1000)
print(f"  Success rate: {final_rate2:.2f}")

env.close()
env2.close()


