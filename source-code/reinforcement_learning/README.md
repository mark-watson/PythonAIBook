# Reinforcement Learning – Source Code

This directory contains example code for the **Overview of Reinforcement Learning** chapter.

## Running

```bash
uv run mdp_demo.py
uv run frozen_lake_qlearning.py
```

## Files

- **mdp_demo.py** — Markov Decision Process examples: a 3×3 grid world solved with value iteration and policy iteration, plus the built-in forest management problem (uses pymdptoolbox)
- **frozen_lake_qlearning.py** — Q-learning agent trained on the Gymnasium FrozenLake environment, with both slippery and deterministic variants

## Architecture

![Reinforcement learning architecture: Q-Learning and MDP approaches](FIG_reinforcement_learning.jpg)
