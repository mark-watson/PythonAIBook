# Overview of Reinforcement Learning (Optional Material)

Reinforcement Learning has been used in various applications such as robotics, game playing, recommendation systems, and more. Reinforcement Learning (RL) is a broad topic and we will only cover basic aspects of RL.

{width: "80%"}
![Architecture diagram for the Reinforcement Learning example](FIG_reinforcement_learning.jpg)

The requirements for this chapter are:

```bash
uv pip install gymnasium numpy pymdptoolbox
```

The examples for this chapter are in the directory **source-code/reinforcement_learning**.

## Overview

Reinforcement Learning is a type of machine learning that is concerned with decision-making in dynamic and uncertain environments. RL uses the concept of an agent which interacts with its environment by taking actions and receiving feedback in the form of rewards or penalties. The goal of the agent is to learn a policy which is a mapping from states of the environment to actions with the goal of maximizing the expected cumulative reward over time.

There are several key components to RL:

- **Environment**: the system or "world" that the agent interacts with.
- **Agent**: the decision-maker that chooses actions based on its current state, the current environment, and its policy.
- **State**: a representation of the current environment, the parameters and trained policy of the agent, and possibly the visible actions of other agents in the environment.
- **Action**: a decision taken by the agent.
- **Reward**: a scalar value that the agent receives as feedback for its actions.

Reinforcement learning algorithms can be divided into two main categories: value-based and policy-based. In value-based RL the agent learns an estimate of the value of different states or state-action pairs which are then used to determine the optimal policy. In contrast, in policy-based RL the agent directly learns a policy without estimating the value of states or state-action pairs.

Reinforcement Learning can be implemented using different techniques such as Q-learning, SARSA, DDPG, A2C, PPO, etc. Some of these techniques are model-based, which means that the agent uses a model of the environment to simulate the effects of different actions and plan ahead. Others are model-free, which means that the agent learns directly from the rewards and transitions experienced in the environment.

If you enjoy the overview material in this chapter I recommend that you consider investing the time in the Coursera RL specialization [https://www.coursera.org/learn/fundamentals-of-reinforcement-learning](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning#instructors) taught by Martha and Adam White. There are [50+ RL courses on Coursera](https://www.coursera.org/courses?query=reinforcement%20learning). I took the courses taught by Martha and Adam White before starting my RL project at Capital One.

My favorite RL book is "Reinforcement Learning: An Introduction, second edition" by Richard Sutton and Andrew Barto, that can be read online for free at [http://www.incompleteideas.net/book/the-book-2nd.html](http://www.incompleteideas.net/book/the-book-2nd.html). They originally wrote their book examples in Common Lisp but most of the code is available rewritten in Python. The Common Lisp code for the examples is [here](http://www.incompleteideas.net/book/code/code2nd.html). Shangtong Zhang translated the book examples to Python, available [here](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction). Martha and Adam White's Coursera class uses this book as a reference.

The core idea of RL is that we train a software agent to interact with and change its environment based on its expectations of the utility of current actions improving metrics for success in the future. There is some tension between writing agents that simply reuse past actions that proved to be useful, rather than aggressively exploring new actions in the environment. There are interesting formalisms for this that we will cover.

There are two general approaches to providing training environments to Reinforcement Learning trained agents: physically devices in the real world or simulated environments. This is not a book on robotics so we use the second option.

The end goal for modeling a RL problem is calculating a policy that can be used to control an agent in environments that are similar to the training environment. In a model at time **t** we have a given State~t~. RL policies can be continually be updated during training and in production environments. A policy given a State~t~, calculates an Action~t~ to execute and changes the state to State~t+1~.

## Available RL Tools

For initial experiments with RL, I would recommend taking the same path that I took before using RL at work:

- Using a maintained fork of OpenAI's Gym library [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
- Taking the Coursera classes by Martha and Adam White.
- The Sutton/Barto RL Book and accompanying Common Lisp or Python examples.

The original OpenAI RL Gym was a good environment for getting started with simple environments and examples but I didn't get very far with self-study. The RL Coursera classes were a great overview of theory and practice, and I then spend as much time as I could spare working through Sutton/Barto before my project started.

## An Introduction to Markov Decision Process

Before we can write a reinforcement learning agent, we need to understand the mathematical framework that RL is built upon: the **Markov Decision Process** (MDP). An MDP provides a formal way to model sequential decision-making problems where outcomes are partly random and partly under the control of a decision-maker.

Let's start by defining the key terms:

- **Sequential decision problem**: a problem where decisions are made in sequence over time, and each decision affects future states and rewards. Unlike one-shot classification or regression, you must think ahead.

- **Fully observable**: the agent can see the complete state of the environment at each step. No hidden information or partial views.

- **Stochastic environment**: transitions between states are not deterministic. An action taken in a given state may lead to different outcomes with certain probabilities. The real world is uncertain, and MDPs model this uncertainty.

- **Markov property**: the future depends only on the current state and action, not on the history of how you got there. Formally, P(State~t+1~ | State~t~, Action~t~) = P(State~t+1~ | State~t~, Action~t~, State~t-1~, Action~t-1~, ...).

- **Bellman equation**: the recursive relationship that expresses the value of a state as the expected immediate reward plus the discounted value of the next state. This is the foundation of dynamic programming in RL: V(s) = max_a [ R(s,a) + γ Σ P(s'|s,a) V(s') ] where γ (gamma) is the discount factor.

The **discount factor** γ (between 0 and 1) controls how much the agent values future rewards versus immediate rewards:
- γ close to 0: agent is shortsighted, cares mostly about immediate reward
- γ close to 1: agent is farsighted, cares about long-term cumulative reward

### Solving MDPs with pymdptoolbox

The [pymdptoolbox](https://github.com/sawcordwell/pymdptoolbox) library provides classic MDP solution algorithms. Let's work through two examples: a custom grid world and the built-in forest management problem.

Listing of **mdp_demo.py**:

```python
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
```

Here is the output of running **mdp_demo.py** (the unicode characters for directionsin the Optimal policy output don't print correctly):

```bash
$ python mdp_demo.py
=======================================================
Markov Decision Process Demo
=======================================================

--- Example 1: Custom 3x3 Grid World ---
Optimal policy:
  →    ↓    ↓  
  →    ↓    ↓  
  →    →    →  

Value function: [ 6.56 13.85 17.45 13.85 21.95 25.95 21.95 30.95 40.95]
Iterations to converge: 5

--- Policy Iteration on same grid ---
Policy: (1, 2, 2, 1, 2, 2, 1, 1, 1)
Iterations: 5

--- Example 2: Forest Management (built-in) ---
States: 5 (forest age classes 0-4)
Action 0 = Wait, Action 1 = Cut
p(fire) = 0.1 each year
Reward shape: (5, 2) (states × actions)
Optimal policy: (0, 0, 0, 0, 0)
  Forest age 0: Wait
  Forest age 1: Wait
  Forest age 2: Wait
  Forest age 3: Wait
  Forest age 4: Wait
Value function: [ 3.49  5.62  8.24 11.48 15.48]
Iterations: 6
```

Walking through **Example 1**, we create a 3×3 grid where each cell is a state (0 through 8). The agent can move up, right, down, or left. The transition matrix `P` has shape `(actions, states, states)` — for each action and current state, it specifies the probability of landing in each next state. Our grid is deterministic (probability of 1.0 for each move), and bumping into walls keeps the agent in place.

The reward matrix `R` gives 10 for reaching the goal (bottom-right, state 8) and -5 for the trap (state 5, which is the middle-right cell). Value iteration converges in 5 iterations and produces a sensible policy: the arrows point around the trap toward the goal. The value function shows that states closer to the goal have higher values.

I also ran **Policy Iteration** on the same grid. Both algorithms arrived at the same policy but through different means: value iteration improves value estimates until convergence, while policy iteration alternates between evaluating a fixed policy and improving it greedily.

In **Example 2**, we use pymdptoolbox's built-in forest management problem. At each year, you choose to Wait (let the forest grow one age class) or Cut (harvest timber, resetting the forest to age 0). There is a 10% chance of fire each year that also resets the forest to age 0. With these parameters the optimal policy is to always Wait — the expected value of older forest outweighs the immediate reward of cutting.

The key takeaway: MDPs give you a formal language to describe decision problems, and algorithms like value iteration and policy iteration compute optimal policies. In the next section, we will tackle environments where we do *not* know the transition probabilities in advance — which is where reinforcement learning comes in.

## A Concrete Example: Q-Learning with Gymnasium

When the transition and reward models are unknown, the agent must learn through trial and error. **Q-learning** is one of the simplest and most widely used model-free RL algorithms. It learns an action-value function Q(s,a) — the expected cumulative reward of taking action **a** in state **s** and following the optimal policy thereafter.

The Q-learning update rule is:

Q(s,a) ← Q(s,a) + α [ r + γ · max_a' Q(s',a') — Q(s,a) ]

Where:
- α (alpha) is the learning rate
- γ (gamma) is the discount factor
- r is the immediate reward received
- s' is the next state

The agent also needs to balance **exploration** (trying random actions to discover better strategies) against **exploitation** (using known good actions). We use an **epsilon-greedy** strategy: with probability ε, take a random action; otherwise take the action with the highest Q-value. Over time we decay ε so the agent explores less and exploits more.

We will use [Gymnasium](https://gymnasium.farama.org/) (the maintained successor to OpenAI Gym) and its **FrozenLake** environment. In FrozenLake, the agent navigates a 4×4 grid of frozen and cracked ice to reach a goal without falling through. By default, the ice is slippery — your intended move only succeeds with 1/3 probability, and you slide perpendicular otherwise.

Listing of **frozen_lake_qlearning.py**:

```python
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
```

Here is the output:

```bash
$ python frozen_lake_qlearning.py
=======================================================
Q-Learning on FrozenLake (slippery)
=======================================================
States: 16
Actions: 4
Map size: 4x4

Training:
  Episode  1000: success rate = 0.68
  Episode  2000: success rate = 0.77
  Episode  3000: success rate = 0.69
  Episode  4000: success rate = 0.75
  Episode  5000: success rate = 0.77
  Episode  6000: success rate = 0.50
  Episode  7000: success rate = 0.77
  Episode  8000: success rate = 0.77
  Episode  9000: success rate = 0.70
  Episode 10000: success rate = 0.75

Learned policy for FrozenLake-v1 (slippery):
  ←↑↑↑
  ←←→←
  ↑↓←←
  ←→↓←
  (←=left ↓=down →=right ↑=up)

Final evaluation (1000 episodes):
  Success rate: 0.74

=======================================================
Q-Learning on FrozenLake (deterministic, no slipping)
=======================================================
States: 16
Actions: 4

Training:
  Episode  1000: success rate = 1.00
  Episode  2000: success rate = 1.00

Learned policy for FrozenLake-v1 (deterministic):
  ↓←←←
  ↓←↓←
  →→↓←
  ←→→←
  (←=left ↓=down →=right ↑=up)

Final evaluation (1000 episodes):
  Success rate: 1.00
```

The `q_learning` function implements the core algorithm. We maintain a Q-table of shape `(n_states, n_actions)` initialized to zeros. Each episode runs until termination (falling in a hole or reaching the goal) or truncation (100-step limit). The epsilon-greedy strategy decays from 1.0 (pure exploration) to 0.01 over time, following the decay schedule `epsilon *= 0.999` each episode.

The Q-value update is the heart of Q-learning:

```python
best_next = np.max(Q[next_state])
Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
```

This says: move Q(s,a) a small step (controlled by alpha) toward `r + γ·max_a' Q(s',a')`. The difference `r + γ·max Q - Q` is called the **temporal-difference error** — how much better or worse the outcome was than predicted.

Let's discuss the results:

**Slippery version**: Achieves ~75% success rate after 10,000 episodes. This is about as good as tabular Q-learning gets on the slippery FrozenLake — the randomness of ice physics means some runs are doomed regardless of policy. The learned policy arrows point toward the goal, though the randomness makes the policy harder to interpret visually than the deterministic case. You can also see the success rate fluctuate (e.g., dropping to 0.50 at episode 6000) — this is normal for Q-learning on stochastic environments and reflects the exploration-exploitation trade-off.

**Deterministic version** (is_slippery=False): Converges to a 100% success rate within 1,000 episodes. Without slipping, the task reduces to a simple shortest-path problem and Q-learning finds it quickly.

### Other Gymnasium Environments Worth Exploring

Once you are comfortable with FrozenLake, Gymnasium offers many environments to experiment with:

- **Taxi-v3**: 500 discrete states, the agent must pick up and drop off a passenger at correct locations. 5 actions (move directions + pickup/dropoff). Good for testing discrete Q-learning at slightly larger scale.
- **CartPole-v1**: balance a pole on a cart. Continuous observation space (4 values), discrete actions (2). Requires function approximation (e.g., a neural network) rather than a Q-table.
- **MountainCar-v0**: drive an underpowered car up a hill. Continuous observations, discrete actions. Classic example where exploration strategy matters deeply.
- **LunarLander-v3**: land a spacecraft on a landing pad. 8 continuous observations, 4 discrete actions. More complex dynamics.

For continuous observation spaces, you cannot use a lookup table — you need **Deep Q-Networks (DQN)** which use neural networks to approximate Q(s,a). That topic is beyond our scope here, but the Sutton/Barto book covers it well.

{class: tip}
When learning RL, start with tabular methods (Q-learning on discrete environments) before moving to deep RL. The concepts transfer directly, but debugging is far easier when you can inspect every Q-value in a table.

## Reinforcement Learning Wrap-up

In this chapter we covered:

- **Markov Decision Processes**: the mathematical foundation of RL, including states, actions, rewards, transition probabilities, the Bellman equation, and discount factors.
- **MDP solving with pymdptoolbox**: value iteration and policy iteration on custom and built-in problems.
- **Q-learning**: a model-free algorithm that learns from experience without needing transition probabilities. We implemented it from scratch and trained agents on the FrozenLake environment.
- **Exploration vs exploitation**: controlled by the epsilon-greedy strategy with decaying epsilon.

If this chapter sparked your interest, I encourage you to work through the Coursera specialization by Martha and Adam White and the Sutton/Barto book. For Python-focused RL, the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) library provides reliable implementations of DQN, A2C, PPO, and other algorithms ready to use with Gymnasium environments.

I tagged this chapter as optional material because I believe most readers will get more immediate value from mastering deep learning and pre-trained models. But if you find yourself working on sequential decision-making problems — robotics, game AI, resource allocation, dynamic pricing — the RL toolkit becomes indispensable.

## Optional Practice Problems

### Problem 1: Parameter Sensitivity in Forest Management (Easy)

In the Forest Management example in [mdp_demo.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/reinforcement_learning/mdp_demo.py), the optimal policy with a discount factor of $\gamma = 0.9$ and a fire probability of $p = 0.1$ is to always Wait.
1. Modify the script to perform a parameter sweep over different discount factors $\gamma \in \{0.1, 0.5, 0.9, 0.99\}$ and fire probabilities $p \in \{0.01, 0.05, 0.1, 0.3, 0.5\}$.
2. Record the resulting optimal policy for each combination.
3. Explain intuitively how a low discount factor (valuing short-term gains) or a high risk of fire (destroying progress) changes the agent's behavior from "Wait" to "Cut".

### Problem 2: Implementing a Stochastic Grid World (Medium)

The 3x3 grid world in [mdp_demo.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/reinforcement_learning/mdp_demo.py) is completely deterministic: when the agent takes an action, it transitions to the target cell with a probability of 1.0.
1. Modify the transition matrix creation in [mdp_demo.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/reinforcement_learning/mdp_demo.py) to model a slippery grid. If the agent chooses a movement action (e.g., move Right), it should succeed with a probability of 0.8. With a probability of 0.1, it slips to the left (perpendicularly), and with a probability of 0.1, it slips to the right. (Note: Bumping into a wall still results in staying in the same cell).
2. Run both Value Iteration and Policy Iteration on this stochastic environment.
3. Compare the resulting value function and optimal policy with the deterministic version. How does transition noise affect the expected cumulative rewards of the non-goal states?

### Problem 3: Scaling Q-Learning to the 8x8 Frozen Lake (Hard)

The tabular Q-learning code in [frozen_lake_qlearning.py](file:///Users/markwatson/GITHUB/PythonAIBook/source-code/reinforcement_learning/frozen_lake_qlearning.py) is configured for the 4x4 `FrozenLake-v1` map.
1. Modify the script to use the larger 8x8 grid: `gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)`.
2. Train the agent using the original parameters. You will likely observe that the success rate remains extremely low (often close to 0) because the reward is sparse (only 1.0 at the goal) and the state space is four times larger.
3. Implement one or both of the following enhancements to solve the 8x8 grid:
   - **Hyperparameter Sweep**: Write a search script to find optimal values for the learning rate $\alpha$, the decay rate of $\epsilon$, and the number of training episodes (hint: you may need 50,000+ episodes).
   - **Reward Shaping**: Implement a custom wrapper or modify the step loop to provide intermediate feedback. For example, give a small positive reward proportional to the decrease in Manhattan distance to the goal, or apply a step penalty (e.g., -0.01) for non-goal steps to discourage circular paths.
4. Report your final success rate and compare it against the baseline.
