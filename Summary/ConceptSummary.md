# Reinforcement Learning Concept Summary

# Table of Contents
1. [Reason Why Failed the Export](#1-reason-why-failed-the-export)
2. [Expensive part in RL](#2-expensive-part-in-rl)
3. [Concept to update policy](#3-concept-to-update-policy)
4. [Types of RL](#4-types-of-rl)
5. [Things to consider when choosing an algorithm](#5-things-to-consider-when-choosing-an-algorithm)
6. [Baselines in policy gradient](#6-baselines-in-policy-gradient)
7. [Importance Sampling](#7-importance-sampling)

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

## 6. Baselines in policy gradient
- Training policy gradient is unstable, in part because the variance of the gradient estimates is large.
    - Large variance means, the $G_t$ value obtained by progressing the episode is sometimes very large and sometimes very small.
    - This is because the progress in each episode is different, and various randomnesses exist in the environment, such as the initial state distribution, policy distribution, and transition probability distribution.
- Baseline
    - $b=\frac{1}{N}\sum_{i=1}^{N}r(\tau)$
- Subtracting a baseline
    - Not effect the policy gradient. (unbiased in expectation)
    <div align="center">
        <img src="./concept_figures/6_1.svg" alt="Equation" style="display: block; margin: 0 auto; background-color: white;">
    </div>
    <!-- $$\begin{align*}
    {\triangledown}_{\theta} J_\theta &\approx \mathbb{E}_{\tau \sim p_\theta(\tau)}[G_0 \ {\triangledown}_{\theta} \ \ln{P_\theta(\tau)}]\\
    \mathbb{E}_{\tau \sim p_\theta(\tau)}[{\triangledown}_{\theta} \ \ln{P_\theta(\tau)} b] &= \int_\tau \ {\triangledown}_{\theta} \ \ln{P_\theta(\tau)} \ P_\theta(\tau) \ b \ d\tau\\
    &= \int_\tau \ {\triangledown}_{\theta} \ P_\theta(\tau) \ b \ d\tau\\
    &= b {\triangledown}_{\theta}  \int_\tau \ P_\theta(\tau) \ d\tau\\
    &= b {\triangledown}_{\theta}  1\\
    &=0 \\
    \end{align*}$$ -->

    - It means:
    <div align="center">
        <img src="./concept_figures/6_2.svg" alt="Equation" style="display: block; margin: 0 auto; background-color: white;">
    </div>
    <!-- $$\begin{align*}
    \mathbb{E}_{\tau \sim p_\theta(\tau)}[{\triangledown}_{\theta} \ \ln{P_\theta(\tau)(G_0 - b)}] = \mathbb{E}_{\tau \sim p_\theta(\tau)}[{\triangledown}_{\theta} \ \ln{P_\theta(\tau)(G_0)}] = {\triangledown}_{\theta} J_\theta\\
    \end{align*}$$ -->

- Subtracting a baseline reduce variance
<div align="center">
    <img src="./concept_figures/6_3.svg" alt="Equation" style="display: block; margin: 0 auto; background-color: white;">
</div>
<!-- $$\begin{align*}
\mathrm{Var_{}} &= \mathbb{E}[x^2] - \mathbb{E}[x]^2 \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}[({\triangledown}_{\theta} \ \ln{P_\theta(\tau)(G_0 - b)})^2] - \mathbb{E}_{\tau \sim p_\theta(\tau)}[{\triangledown}_{\theta} \ \ln{P_\theta(\tau)(G_0 - b)}]^2 \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)}[({\triangledown}_{\theta} \ \ln{P_\theta(\tau)(G_0 - b)})^2] - \mathbb{E}_{\tau \sim p_\theta(\tau)}[{\triangledown}_{\theta} \ \ln{P_\theta(\tau)G_0}]^2 \\
\frac{d\mathrm{Var}}{db} &= \frac{d}{db} \mathbb{E}_{\tau \sim p_\theta(\tau)}[({\triangledown}_{\theta} \ \ln{P_\theta(\tau)(G_0 - b)})^2] \\
&= \frac{d}{db} \mathbb{E}_{\tau \sim p_\theta(\tau)}[(g(\tau)(G_0 - b))^2] \\
&= -2\mathbb{E}_{\tau \sim p_\theta(\tau)}[g(\tau)^2G_0] + 2b\mathbb{E}_{\tau \sim p_\theta(\tau)}[g(\tau)^2]\\
b&=\frac{\mathbb{E}_{\tau \sim p_\theta(\tau)}[g(\tau)^2G_0]}{\mathbb{E}_{\tau \sim p_\theta(\tau)}[g(\tau)^2]} \\
\end{align*}$$ -->

## 7. Importance Sampling
- Use the sample from $\bar{P}$ to learn while my RL policy is now $P$.
- Convert on-policy RL to off-policy RL.
<div align="center">
    <img src="./concept_figures/7_1.svg" alt="Equation" style="display: block; margin: 0 auto; background-color: white;">
</div>
<!-- $$\begin{align*}
\mathbb{E}_{x \sim p(x)}[f(x)] &= \int p(x)f(x)dx\\
&= \int \frac{\bar{p}(x)}{\bar{p}(x)}p(x)f(x)dx\\
&= \int \bar{p}(x)\frac{p(x)}{\bar{p}(x)}f(x)dx\\
&= \mathbb{E}_{x \sim \bar{p}(x)} \biggl[ \frac{p(x)}{\bar{p}(x)}f(x) \biggr] \\
\end{align*}$$ -->

- In policy gradient,
<div align="center">
    <img src="./concept_figures/7_2.svg" alt="Equation" style="display: block; margin: 0 auto; background-color: white;">
</div>
<!-- $$\begin{align*}
\frac{P_\theta(\tau)}{\bar{P}(\tau)}&=\frac{P(s_1)\prod_{t=1}^T P_\theta(a_t \mid s_t) P(s_{t+1}\mid s_t,a_t)}{P(s_1)\prod_{t=1}^T \bar{P}(a_t \mid s_t) P(s_{t+1}\mid s_t,a_t)}\\
&\approx \frac{\prod_{t=1}^T P_\theta(a_t \mid s_t)}{\prod_{t=1}^T \bar{P}(a_t \mid s_t)}\\
{\triangledown}_{\theta}J_\theta&=\mathbb{E}_{\tau \sim p_\theta} [G_0 \ {\triangledown}_{\theta} \ \ln{P_\theta(\tau)}]\\
&=\mathbb{E}_{\tau \sim \bar{p}} \biggl[ \frac{p_\theta(\tau)}{\bar{p}(\tau)} \ G_0 \ {\triangledown}_{\theta} \ \ln{P_\theta(\tau)} \biggr]\\
&\approx \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T \frac{ P_\theta(a_t \mid s_t)}{ \bar{P}(a_t \mid s_t)} \ G_0 \ {\triangledown}_{\theta} \ \ln{P_\theta(a_{i,t} \mid s_{i,t})}
\end{align*}$$ -->

