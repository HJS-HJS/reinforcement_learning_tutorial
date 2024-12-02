# reinforcement_learning_tutorial
<div align="center">
    <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
</div>

- A repository for studing reinforcement learning using Cartpole.

# Table of Contents

1. [Setup](#1-setup)
2. [Tutorial](#2-tutorial)
    1. [Check environments](#1-check-environments)
    2. [Simple policy](#2-simple-policy)
3. [Value-Based](#3-value-based)
    1. [Q-learning](#1-q-learning)
    2. [DQN](#2-dqn)
4. [Policy_based](#4-policy-based)
    1. [REINFORCE](#1-reinforce)
    2. [#TODO A2C](#2-a2c)
    3. [#TODO PPO](#3-ppo)
    4. [#TODO SAC](#4-sac)

## 1. Setup

1. **Download tutorial code**
   ```bash
   mkdir ~/reinforcement_learning_tutorial
   git clone https://github.com/HJS-HJS/reinforcement_learning_tutorial.git reinforcement_learning_tutorial
   ```

2. **Install required python library**
   ```bash
   cd ~/reinforcement_learning_tutorial
   pip3 install -r requirements.txt
   ```

## 2. Tutorial 

### 1. Check environments
- Check if CUDA is properly installed
    ```bash
    python3 ~/reinforcement_learning_tutorial/tutorial1_1_check_cuda.py
    ```
- Check if cartpole is running
    ```bash
    python3 ~/reinforcement_learning_tutorial/tutorial1_2_start_gym.py
    ```

### 2. Simple policy
- An example of moving a cartpole with an unchanging policy.
    ```bash
    python3 ~/reinforcement_learning_tutorial/tutorial2_start_policy.py
    ```

## 3. Value-Based
### 1. Q-learning
- Reinforcement learning wth Q table
    ```bash
    python3 ~/reinforcement_learning_tutorial/value_based_1_1_q_learning.py
    ```
- Reinforcement learning wth Q-learning (with epsilon greedy)
    ```bash
    python3 ~/reinforcement_learning_tutorial/value_based_1_2_q_learning_greedy.py
    ```

### 2. DQN
- paper
    - [Playing Atari with Deep Reinforcement Learning (2013)](https://arxiv.org/pdf/1312.5602)
    - [Human-level control through deep reinforcement learning (2015)](https://www.nature.com/articles/nature14236)
- Reinforcement learning wth DQN
    ```bash
    python3 ~/reinforcement_learning_tutorial/value_based_2_dqn.py
    ```

## 4. Policy-Based
### 1. REINFORCE
- Reinforcement learning wth REINFORCE
    ```bash
    python3 ~/reinforcement_learning_tutorial/policy_based_1_REINFORCE.py
    ```

### 2. A2C
- TODO

### 3. PPO
- TODO

### 4. SAC
- TODO
