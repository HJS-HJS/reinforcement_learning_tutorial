'''
PPO (Proximal Policy Optimization)
- continous action
- multiprocessing
'''
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp  # 멀티프로세싱 모듈 추가

from utils.sac_multiprocess               import SAC
from utils.half_cheetah      import HalfCheetah
from utils.utils import str2bool

# Network model
class ActorNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_action:int = 2):
        super(ActorNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.mu = nn.Sequential(
            nn.Linear(256,n_action),
        )
        self.std = nn.Sequential(
            nn.Linear(256,n_action),
            nn.Softplus(),
        )

    def forward(self, state):
        x = self.layer(state)
        mu = self.mu(x)
        std = self.std(x)

        # sample
        distribution = torch.distributions.Normal(mu, std)
        u = distribution.rsample()
        logprob = distribution.log_prob(u)

        # Enforce action bounds [-1., 1.]
        action = torch.tanh(u)
        logprob = logprob - torch.log(1 - torch.tanh(u).pow(2) + 1e-7)

        return action, logprob

    def normalize_action(self, action:torch.tensor):
        return action.squeeze().cpu().numpy()
    
class QNetwork(nn.Module):
    def __init__(self, n_state:int = 4, n_action:int = 2):
        super(QNetwork, self).__init__()
        self.state_layer = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
        )
        self.action_layer = nn.Sequential(
            nn.Linear(n_action, 128),
            nn.ReLU(),
        )
        self.layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        _state = self.state_layer(state)
        _action = self.action_layer(action)
        return self.layer(torch.cat([_state, _action], dim=1))

## Parameters
# Learning Parameters
LEARNING_RATE   = 0.0005
ALPHA_LR        = 0.001
DISCOUNT_FACTOR = 0.99
TAU             = 0.005
ALPHA           = 0.01
TARGET_ENTROPY  = -6.0
# Memory & MP Settings
EPISODES        = 20000
MAX_STEP        = 1024
MEMORY_CAPACITY = 100000
BATCH_SIZE      = 1024
EPOCH_SIZE      = 10
# Other
visulaize_step  = 25
SAVE_DIR = "./model/tutorial_continuos_2_3_SAC_multi"
FILE_NAME = "1275_actor_critic"

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
                           default=4)

    args = argparser.parse_args()

    TRAIN = args.Train
    NUM_WORKERS = args.Worker

    # 멀티프로세싱 시작 방식 설정
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dummy_sim = HalfCheetah(None)
    N_INPUTS = dummy_sim.N_INPUTS
    N_OUTPUT = dummy_sim.N_OUTPUT
    dummy_sim.env.close()

    # SAC Agent 초기화
    sac_agent = SAC(
        actor_net_cls=ActorNetwork,
        q_net_cls=QNetwork,
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUT,
        workers=NUM_WORKERS,
        environment=HalfCheetah,
        batch_size=BATCH_SIZE,
        max_steps=MAX_STEP,
        max_episode=EPISODES,
        memory_capacity=MEMORY_CAPACITY,
        learning_rate=LEARNING_RATE,
        epoch=EPOCH_SIZE,
        alpha_lr=ALPHA_LR,
        discount_factor=DISCOUNT_FACTOR,
        tau=TAU,
        alpha=ALPHA,
        target_entropy=TARGET_ENTROPY,
        device=device,
        save_dir=SAVE_DIR,
        save_period=visulaize_step,
    )

    # 학습 시작
    if TRAIN:
        sac_agent.train()
    else:
        sac_agent.execute(FILE_NAME, BATCH_SIZE)

