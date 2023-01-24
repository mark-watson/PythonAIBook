# Reinforcement Learning

Reinforcement Learning (RL) is a broad topic and we will only cover aspects RL that I use myself.

{class: information}
This is a common theme in this book: if I don't love a topic or I don't have much practical experience with it, I generally don't write about it or cover it lightly with references for further study. I used RL a few years ago on a project at Capital One and here I am guiding you on the same learning path that I took prior to working on that project.

If you enjoy the tutorial material in this chapter I recommend that you consider investing the time in the Coursera RL specialization [https://www.coursera.org/learn/fundamentals-of-reinforcement-learning](https://www.coursera.org/learn/fundamentals-of-reinforcement-learning#instructors) taught by Martha and Adam White. There are [50+ RL courses on Coursera](https://www.coursera.org/courses?query=reinforcement%20learning). I took the courses taught by Martha and Adam White before starting my RL project at Capital One.

My favorite RL book is "Reinforcement Learning: An Introduction, second edition" by Richard Sutton and Andrew Barto, that can be read online for free at [http://www.incompleteideas.net/book/the-book-2nd.html](http://www.incompleteideas.net/book/the-book-2nd.html). They origianlly wrote their book examples in Common Lisp but most of the code is available rewritten in Python. The Common Lisp code for the examples is [here](http://www.incompleteideas.net/book/code/code2nd.html). Shangtong Zhang translated the book examples to Python, available [here](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction). Martha and Adam White's Coursera class uses this book as a reference.


## Overview

The core idea of RL is that we train a software agent to interact with and change its environment based on its expectations of the utility of current actions improving metrics for success in the future. There is some tension between writing agents that simply reuse past actions that proved to be useful, rather than aggresively exploring new actions in the environment. There are intersting formalisms for this that we will cover.

There are two general approaches to providing training environments to Reinforcement Learning trained agents: physically devices in the real world or simulated environments. This is not a book on robotics so we use the second option.

The end goal for modeling a RL problem is calculating a policy that can be used to control an agent in environments that are similar to the training environment. In a model at time **t** we have a given State~t~. RL policies can be continually be updated during training and in production environments. A policy given a State~t~, calculates an Action~t~ to execute and change the state to State~t+1~.

## Available RL Tools

