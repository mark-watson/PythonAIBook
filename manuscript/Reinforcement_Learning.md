# Overview of Reinforcement Learning (Optional Material)

Reinforcement Learning has been used in various applications such as robotics, game playing, recommendation systems, etc. Reinforcement Learning (RL) is a broad topic and we will only cover aspects RL that I use myself.

We will start with suggested paths of study and end with an introduction to Markov Decision Process and then build on that with a concrete RL example.

{class: information}
This is a common theme in this book: if I don't love a topic or I don't have much practical experience with it, I generally don't write about it or cover it lightly with references for further study. I have limited experience using RL professionally, mostly for a project a few years ago at Capital One and here I am guiding you on the same learning path that I took prior to working on that project.

## Overview

Reinforcement Learning is a type of machine learning that is concerned with decision-making in dynamic and uncertain environments. RL uses the concept of an agent which interacts with its environment by taking actions and receiving feedback in the form of rewards or penalties. The goal of the agent is to learn a policy which is a mapping from states of the environment to actions with the goal of maximizing the expected cumulative reward over time.

There are several key components to RL:

- Environment: the system or "world" that the agent interacts with.
- Agent: the decision-maker that chooses actions based on its current state, the current environment, and its policy.
- State: a representation of the current environment, the parameters and trained policy of the agent, and possibly the visible actions of other agents in the environment.
- Action: a decision taken by the agent.
- Reward: a scalar value that the agent receives as feedback for its actions.

Reinforcement learning algorithms can be divided into two main categories: value-based and policy-based. In value-based RL the agent learns an estimate of the value of different states or state-action pairs which are then used to determine the optimal policy. In contrast, in policy-based RL the agent directly learns a policy without estimating the value of states or state-action pairs.

Reinforcement Learning can be implemented using different techniques such as Q-learning, SARSA, DDPG, A2C, PPO, etc. Some of these techniques are model-based, which means that the agent uses a model of the environment to simulate the effects of different actions and plan ahead. Others are model-free, which means that the agent learns directly from the rewards and transitions experienced in the environment.

If you enjoy the overview material in this chapter I recommend that you consider investing the time in the Coursera RL specialization [https://www.coursera.org/learn/fundamentals-of-reinforcement-learning](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning#instructors) taught by Martha and Adam White. There are [50+ RL courses on Coursera](https://www.coursera.org/courses?query=reinforcement%20learning). I took the courses taught by Martha and Adam White before starting my RL project at Capital One.

My favorite RL book is "Reinforcement Learning: An Introduction, second edition" by Richard Sutton and Andrew Barto, that can be read online for free at [http://www.incompleteideas.net/book/the-book-2nd.html](http://www.incompleteideas.net/book/the-book-2nd.html). They originally wrote their book examples in Common Lisp but most of the code is available rewritten in Python. The Common Lisp code for the examples is [here](http://www.incompleteideas.net/book/code/code2nd.html). Shangtong Zhang translated the book examples to Python, available [here](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction). Martha and Adam White's Coursera class uses this book as a reference.

The core idea of RL is that we train a software agent to interact with and change its environment based on its expectations of the utility of current actions improving metrics for success in the future. There is some tension between writing agents that simply reuse past actions that proved to be useful, rather than aggressively exploring new actions in the environment. There are interesting formalisms for this that we will cover.

There are two general approaches to providing training environments to Reinforcement Learning trained agents: physically devices in the real world or simulated environments. This is not a book on robotics so we use the second option.

The end goal for modeling a RL problem is calculating a policy that can be used to control an agent in environments that are similar to the training environment. In a model at time **t** we have a given State~t~. RL policies can be continually be updated during training and in production environments. A policy given a State~t~, calculates an Action~t~ to execute and changes the state to State~t+1~.

## Available RL Tools

For initial experiments with RL, I would recommend taking the same path that I took before using RL at work:

- Using a maintained fork of OpenAI’s Gym library [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
- Taking the Coursera classes by Martha and Adam White.
- The Sutton/Barto RL Book and accompanying Common Lisp or Python examples.

The original OpenAI RL Gym was a good environment for getting started with simple environments and examples but I didn't get very far with self-study. The RL Coursera classes were a great overview of theory and practice, and I then spend as much time as I could spare working through Sutton/Barto before my project started.

## An Introduction to Markov Decision Process

Here we learn the basic ideas of Markov Decision Process (MDP) and look at a few examples using a popular Python MDP library.

Let's start with defining a few terms you will need to know:

- Sequential decision problem:
- Observable:
- Stochastic environment: 
- Bellman equation: 
-   

TBD

TBD: use the excellent library https://github.com/sawcordwell/pymdptoolbox

## A Concrete Example Implementing RL

TBD: build on the last section on MDP


## Reinforcement Learning Wrap-up

Dear reader, please pardon the brevity of this overview chapter. I may re-work this chapter with a few examples in the next edition of this book. I tagged this chapter as optional material because I feel that most readers will be better off investing limited learning time in understanding how to use deep learning and pre-trained models.
