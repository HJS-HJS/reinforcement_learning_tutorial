'''
PPO (Proximal Policy Optimization)
- continous action
- multiprocessing
'''
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from utils.ppo_multiprocess import PPO
from utils.car_racing       import CarRacing
from utils.utils import str2bool

# Network model
class ActorCriticNetwork(nn.Module):
    def __init__(self, n_state, n_action):
        super(ActorCriticNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=n_state, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128,n_action)
        self.std = nn.Linear(128,n_action)
        self.critic = nn.Linear(128,1)

    def Actor(self, state):
        x = self.layer(state)
        mu = torch.tanh(self.mu(x))
        std = nn.functional.softplus(self.std(x))
        dist = torch.distributions.MultivariateNormal(mu, torch.diag_embed(std, 0))
        return dist
    
    def Critic(self, state):
        x = self.layer(state)
        return self.critic(x)

    def get_action(self, state, action=None):
        distribution = self.Actor(state)

        if action is None:
            action = distribution.sample()

        return action, distribution.log_prob(action)

    def get_mean_action(self, state):
        action = self.Actor(state).mean
        return self.normalize_action(action)

    def normalize_action(self, action:torch.tensor):
        return action.squeeze().cpu().numpy() * np.array([1., 0.5, 0.5]) + np.array([0., 0.5, 0.5])

## Parameters
# Learning Parameters
LEARNING_RATE   = 0.0005
DISCOUNT_FACTOR = 0.99
ADVANTAGE_LAMBDA= 0.95
EPISODES        = 1000
# Memory & MP Settings
BATCH_SIZE      = 256   # 각 워커가 한 번에 수집할 데이터 크기
EPOCH_SIZE      = 3
CLIP_EPSILON    = 0.1
# Other
visulaize_step  = 10
SAVE_DIR = "./model/tutorial_continuos_1_4_PPO_multi"
FILE_NAME = "440_actor_critic"

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-t', 
                           '--Train', 
                           type=str2bool,
                           choices=[True, False],
                           default=False)
    argparser.add_argument('-w', 
                           '--Worker', 
                           type=int,
                           default=12)

    args = argparser.parse_args()
    NUM_WORKERS = args.Worker

    TRAIN = args.Train
    NUM_WORKERS = args.Worker

    # 멀티프로세싱 시작 방식 설정
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_sim = CarRacing(None)
    N_INPUTS = dummy_sim.N_INPUTS
    N_OUTPUT = dummy_sim.N_OUTPUT
    dummy_sim.env.close()

    ppo = PPO(
        actor_critic_net = ActorCriticNetwork, 
        n_inputs= N_INPUTS, 
        n_outputs = N_OUTPUT, 
        workers=NUM_WORKERS,
        environment = CarRacing,
        epoch=EPOCH_SIZE,
        batch=BATCH_SIZE,
        max_episode=EPISODES,
        learning_rate=0.0005,
        discount_factor = DISCOUNT_FACTOR,
        advantage_lambda = ADVANTAGE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        device=device,
        save_dir=SAVE_DIR,
        save_period=visulaize_step,
        )

    if TRAIN:
        ppo.learn()
    else:
        ppo.execute(FILE_NAME, BATCH_SIZE)
