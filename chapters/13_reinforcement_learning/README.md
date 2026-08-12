# Chapter 13: Reinforcement Learning - From MDPs to Deep RL

## Chapter Overview
Unlike supervised learning, Reinforcement Learning (RL) agents do not receive explicit labels; they learn by interacting with an environment to maximize a delayed, scalar reward signal[cite: 1]. This chapter covers the theoretical framework of Markov Decision Processes (MDPs) and scales up to modern Deep RL algorithms.

## Learning Objectives
* Define Markov Decision Processes, the credit assignment problem, and the exploration-exploitation tradeoff[cite: 1].
* Derive the Bellman Expectation and Optimality Equations[cite: 1].
* Implement Temporal Difference (TD) learning algorithms such as Q-Learning and SARSA[cite: 1].
* Understand Deep Q-Networks (DQN) and the necessity of experience replay and target networks[cite: 1].
* Analyze Proximal Policy Optimization (PPO) and its clipped surrogate objective[cite: 1].

## Concepts Covered
* Agent-Environment Loop
* Value Iteration and Policy Iteration
* Q-Learning (Off-Policy TD Control)
* Deep Q-Networks (DQN)
* Policy Gradients and REINFORCE
* Proximal Policy Optimization (PPO)
* Reinforcement Learning from Human Feedback (RLHF)

## Connection to Textbook
The Q-Learning implementation directly utilizes the Bellman Optimality Equation derived in the text to iteratively update state-action values[cite: 1]. We also structure the theoretical notes to cover the three phases of RLHF: Pre-training, Supervised Fine-Tuning, and PPO Optimization[cite: 1].
