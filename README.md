# reinforcement_learning_tutorial
<img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/>
<img alt="Gymnasium" src="https://raw.githubusercontent.com/Farama-Foundation/Gymnasium/main/gymnasium-text.png" width="100px" />

- A repository for studing reinforcement learning.
- All algorithm descriptions are available at the following [link](./Summary/AlgorithmSummary.md).
- All concept descriptions are available at the following [link](./Summary/ConceptSummary.md).

# Table of Contents

1. [Code Running](#1-code-running)
2. [Setup](#2-setup)
3. [Before Start](#3-before-start)
    1. [Check environments](#1-check-environments)
    2. [Simple policy](#2-simple-policy)

## 1. Code Running
| No  | RL           |Action Space| Simulation   | Code                               | Note   |
|-----|--------------|------------|--------------|------------------------------------|-----------|
| 1   | Q-Learning   | Descrete   | Cartpole     | tutorial_1_1_q_learning.py         |  |
|     |              | Descrete   | Cartpole     | tutorial_1_2_q_learning_greedy.py  | $\epsilon$-greedy |
| 2   | DQN          | Descrete   | Cartpole     | tutorial_2_DQN.py                  |  |
| 3   | REINFORCE    | Descrete   | Cartpole     | tutorial_3_REINFORCE.py            |  |
| 4   | Actor-Critic | Descrete   | Cartpole     | tutorial_4_1_actor_critic.py       |  |
|     |              | Descrete   | Cartpole     | tutorial_4_2_actor_critic_share.py | actor and critic share the network |
| 5   | A2C          | Descrete   | Cartpole     | tutorial_5_A2C.py                  |  |
| 6   | PPO          | Descrete   | Cartpole     | tutorial_6_1_PPO.py                |  |
|     |              | Descrete   | Cartpole     | tutorial_6_2_PPO_share.py          | actor and critic share the network |
|     |              | Continous  | Half-Cheetah | tutorial_continuos_1_1_PPO.py      |  |
|     |              | Continous  | Half-Cheetah | tutorial_continuos_1_2_PPO.py      | actor and critic share the network |
|     |              | Continous  | Half-Cheetah | tutorial_continuos_1_3_PPO_multi.py| share network / multiprocessing |
|     |              | Continous  | Car-Racing   | tutorial_continuos_1_4_PPO_multi.py| CNN / share network / multiprocessing |
| 7   | SAC          | Continous  | Half-Cheetah | tutorial_continuos_2_1_SAC.py      |  |
|     |              | Continous  | Car-Racing   | tutorial_continuos_2_2_SAC.py      | CNN |

## 2. Setup

### 1. **Download tutorial code**
```bash
mkdir ~/reinforcement_learning_tutorial
git clone https://github.com/HJS-HJS/reinforcement_learning_tutorial.git reinforcement_learning_tutorial
```

### 2. **Install required library**
```bash
sudo apt-get install patchelf
```

### 3. **Install required python library**
```bash
cd ~/reinforcement_learning_tutorial
pip3 install -r requirements.txt
```

### 4. **Install MUJOCO 2.1.0**
- Download the "mujoco210-linux-x86_64.tar.gz" from [link](https://github.com/google-deepmind/mujoco/releases/tag/2.1.0)
    ```bash
    cd ~/Downloads
    tar -zxvf mujoco210-linux-x86_64.tar.gz
    mkdir ~/.mujoco
    cp -r mujoco210 ~/.mujoco/
    rm -rf mujoco210 mujoco210-linux-x86_64.tar.gz
    echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rise/.mujoco/mujoco210/bin:/usr/lib/nvidia' >> ~/.bashrc
    ```
- Change the variable name from "solver_iter" to "solver_niter"
    ```bash
    gedit {path_to_your_gym_library}/envs/mujoco/mujoco_rendering.py
    ```
- from
    ```bash
        self.add_overlay(
            bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
        )
    ```
- to
    ```bash
        self.add_overlay(
            bottomleft, "Solver iterations", str(self.data.solver_niter + 1)
        )
    ```

## 3. Before Start
### 1. Check environments
- Check if CUDA is properly installed
    ```bash
    python3 ~/reinforcement_learning_tutorial/before_start_1_1_check_cuda.py
    ```
- Check if cartpole is running
    ```bash
    python3 ~/reinforcement_learning_tutorial/before_start_1_2_start_gym.py
    ```

### 2. Simple policy
- An example of moving a cartpole with an unchanging policy.
    ```bash
    python3 ~/reinforcement_learning_tutorial/before_start_2_start_policy.py
    ```

