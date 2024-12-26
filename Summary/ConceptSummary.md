# Reinforcement Learning Concept Summary

# Table of Contents
1. [Reason Why Failed the Export](#1-reason-why-failed-the-export)
2. [Expensive part in RL](#2-expensive-part-in-rl)
3. [Concept to update policy](#3-concept-to-update-policy)
4. [Types of RL](#4-types-of-rl)
5. [Things to consider when choosing an algorithm](#5-things-to-consider-when-choosing-an-algorithm)

## 1. Reason Why Failed the Export
- Causal confusion
    - If cause and effect are not clear and there is room for confusion, learning may be difficult.
- Non-Markovian behavior
    - What happened in the past affects the present
    - Use sequence model (Transformers, LSTM cells, ...)
- Multimodal behavior
    - When the number of possible actions in a situation is diverse
    - Mixture of Gaussians $\sum_i w_i \mathcal{N}(\mu_i , \Sigma_i)$
    - Latent variable models
    - Diffusion models

## 2. Expensive part in RL
- Generate samples (generate train data)
    - It is much cheaper to obtain data through simulation.
- Fit a model (calculate the expected return)
    - Learning the entire model is expensive.
    - Reward-based policy learning is inexpensive.

## 3. Concept to update policy
1. If we have policy $\pi$, and we konw $Q(s,a)$, then we can improve $\pi$.
2. Compute gradient to increase probability of good action a (policy gradient)

## 4. Types of RL
- Value-based
    - Q-learning, DQN, Temporal difference learning, Fitted value iteration
    - Estimate value function or Q-function of the optimal policy (no explicit policy).
- Policy Gradients 
    - REINFORCE, Natural policy gradient, Trust region policy optimization
- Actor-critic
    - Asynchronous advantage actor-critic (A3C), Soft actor-critic (SAC)
    - Estimate value function or Q-function of the current policy, use it to improve policy.
- Model-Based RL
    - Dyna, Guided policy search 
    - Estimate the transition model, and
        - Use it for planning (no explicit policy).
        - Use it to improve a policy.
    - Learn physics how environment work.
    - Learn model for trajectory optimization/ optimal control.
    - Use the model to learn a value function.

## 5. Things to consider when choosing an algorithm
- Sample efficency (on, off polocy)
- Stability (Converge)
- Stochastic or deterministic
- Continuous or discrete
- Episodic or infinite
- Easier to represent the policy
- Easier to represent the model


