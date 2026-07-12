# Q-Learning on FrozenLake
# Demonstrates model-free reinforcement learning using the Q-learning algorithm
# on the classic FrozenLake grid-world environment.
#
# References:
#   Q-learning:  https://en.wikipedia.org/wiki/Q-learning
#   FrozenLake:  https://gymnasium.farama.org/environments/toy_text/frozen_lake/
#   Gymnasium:   https://gymnasium.farama.org/
#   TD learning: https://en.wikipedia.org/wiki/Temporal_difference_learning

import gymnasium as gym  # OpenAI Gymnasium RL environment library
import numpy as np  # Numerical computing with arrays

np.random.seed(42)  # Reproducible results across runs


def q_learning(
    env,
    episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    min_epsilon=0.01,
):
    """Train a Q-table via tabular Q-learning (Watkins, 1989).

    Q-learning is a model-free, off-policy temporal-difference (TD) algorithm.
    It learns an estimate Q(s,a) of the expected return from taking action a
    in state s and thereafter following the optimal policy.

    Parameters:
        env            -- Gymnasium environment instance
        episodes       -- number of training episodes (default 10,000)
        alpha          -- learning rate; how quickly new info overrides old
        gamma          -- discount factor; weight of future vs immediate reward
        epsilon        -- initial exploration rate (1.0 = 100% random actions)
        epsilon_decay  -- multiplicative decay applied each episode
        min_epsilon    -- floor value so exploration never fully stops

    Returns:
        Q              -- learned Q-table of shape (n_states, n_actions)
        success_history -- list of (episode_no, success_rate) tuples

    Further reading:
        https://link.springer.com/article/10.1007/BF00992698       (original paper)
        https://en.wikipedia.org/wiki/Q-learning
    """
    n_states = env.observation_space.n  # discrete state count (16 for 4×4 grid)
    n_actions = env.action_space.n  # discrete action count (4: ←↓→↑)
    Q = np.zeros((n_states, n_actions))  # Q-table initialized to zero
    success_history = []

    for episode in range(episodes):
        state, _ = env.reset()  # start a fresh episode
        done = False

        while not done:
            # Epsilon-greedy exploration:
            # With probability ε, take a random action (explore).
            # Otherwise, exploit by choosing the best-known action.
            # See: https://en.wikipedia.org/wiki/Reinforcement_learning#Exploration
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # explore: random action
            else:
                action = np.argmax(Q[state])  # exploit: best known action

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # episode ends on goal, hole, or time limit

            # Q-learning update (TD(0) rule):
            # Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
            # The target r + γ·max_b Q(next,b) is an off-policy bootstrap estimate.
            best_next = np.max(Q[next_state])  # max over actions in next state
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state

        # Decay epsilon so the agent explores less as it learns more
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Log progress every 1,000 episodes
        if (episode + 1) % 1000 == 0:
            success_rate = evaluate(env, Q, episodes=100)
            success_history.append((episode + 1, success_rate))
            print(f"  Episode {episode + 1:>5}: success rate = {success_rate:.2f}")

    return Q, success_history


def evaluate(env, Q, episodes=100):
    """Evaluate a greedy policy derived from Q-table over multiple episodes.

    Runs the agent with zero exploration (argmax over Q row), counting how
    many episodes end at the goal (reward == 1.0). Returns success ratio.

    Parameters:
        env      -- Gymnasium environment
        Q        -- learned Q-table
        episodes -- number of evaluation rollouts (default 100)

    Returns:
        float -- success rate in [0, 1]
    """
    successes = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])  # greedy: always pick best action
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward == 1.0:  # reached the goal tile
                successes += 1
    return successes / episodes


def print_policy(Q, env_name):
    """Print the greedy policy as a grid of arrow characters.

    Maps each state's argmax action to a Unicode arrow and displays it as a
    4×4 grid matching the FrozenLake layout.

    Action mapping: 0=← (left), 1=↓ (down), 2=→ (right), 3=↑ (up)
    """
    n_states = Q.shape[0]
    size = int(np.sqrt(n_states))  # assume square grid (4 for FrozenLake)
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    print(f"\nLearned policy for {env_name}:")
    for row in range(size):
        row_str = ""
        for col in range(size):
            s = row * size + col  # flatten 2D grid index to state index
            row_str += arrows[np.argmax(Q[s])]  # best action for this state
        print(f"  {row_str}")
    print("  (←=left ↓=down →=right ↑=up)")


# =============================================================================
# Experiment 1: FrozenLake with slippery ice (stochastic transitions)
# In this mode, the agent only moves in the chosen direction 1/3 of the time;
# the other 2/3 it slides perpendicularly. This matches the classic benchmark.
# =============================================================================

print("=" * 55)
print("Q-Learning on FrozenLake (slippery)")
print("=" * 55)

# Create environment: 4×4 grid, stochastic transitions
# See: https://gymnasium.farama.org/environments/toy_text/frozen_lake/
env = gym.make("FrozenLake-v1", is_slippery=True)
print(f"States: {env.observation_space.n}")
print(f"Actions: {env.action_space.n}")
print("Map size: 4x4")

print("\nTraining:")
Q, history = q_learning(env)

print_policy(Q, "FrozenLake-v1 (slippery)")

# Evaluate the final policy over 1000 rollout episodes
print("\nFinal evaluation (1000 episodes):")
final_rate = evaluate(env, Q, episodes=1000)
print(f"  Success rate: {final_rate:.2f}")

# =============================================================================
# Experiment 2: FrozenLake with deterministic ice (no slipping)
# Disabling slippage makes the environment fully predictable, so the Q-learning
# agent converges much faster and to a higher success rate.
# =============================================================================

print("\n" + "=" * 55)
print("Q-Learning on FrozenLake (deterministic, no slipping)")
print("=" * 55)

env2 = gym.make("FrozenLake-v1", is_slippery=False)
print(f"States: {env2.observation_space.n}")
print(f"Actions: {env2.action_space.n}")

print("\nTraining:")
Q2, _ = q_learning(env2, episodes=2000)  # fewer episodes needed for deterministic

print_policy(Q2, "FrozenLake-v1 (deterministic)")

print("\nFinal evaluation (1000 episodes):")
final_rate2 = evaluate(env2, Q2, episodes=1000)
print(f"  Success rate: {final_rate2:.2f}")

env.close()
env2.close()
