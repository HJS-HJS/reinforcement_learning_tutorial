# reinforcement_learning_tutorial
<div align="center">
    <img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
</div>

- A repository for studing reinforcement learning using Cartpole.

# Table of Contents

1. [Setup](#1-setup)
2. [Tutorial](#2-tutorial)

    0. [Check environments](#0-check-environments)
        1. Check CUDA
        2. Check cartpole
    1. [Simple policy](#1-simple-polisy)
    2. [#TODO Q-learning](#2-q-learning)
    3. [#TODO DQN](#-)
    4. [#TODO DDQN](#-)
    5. [#TODO SARSA](#-)
    6. [#TODO PPO](#-)

## 1. Setup

1. **Download tutorial code**
   ```bash
   cd
   mkdir reinforcement_learning_tutorial
   git clone https://github.com/HJS-HJS/reinforcement_learning_tutorial.git reinforcement_learning_tutorial
   ```

2. **Install required python library**
   ```bash
   cd ~/reinforcement_learning_tutorial
   pip3 install -r requirements.txt
   ```

## 2. Tutorial 

### 0. Check environments
1. Check CUDA
    - Check if CUDA is properly installed
        ```bash
        cd ~/reinforcement_learning_tutorial
        python3 tutorial0_check_cuda.py
        ```
2.  Check cartpole
    - Check if cartpole is running
        ```bash
        cd ~/reinforcement_learning_tutorial
        python3 tutorial0_start_gym.py
        ```

### 1. Simple policy
- An example of moving a cartpole with an unchanging policy.
    ```bash
    cd ~/reinforcement_learning_tutorial
    python3 tutorial1_start_policy.py
    ```

### 2. Q-learning
- Reinforcement learning using Q-learning (not using a neural network)
    ```bash
    cd ~/reinforcement_learning_tutorial
    python3 tutorial2_q_learning.py
    ```