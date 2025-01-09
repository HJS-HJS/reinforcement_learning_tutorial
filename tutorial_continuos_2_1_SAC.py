'''
SAC (Soft Actor Critic)
- continous action
'''

import os
import numpy as np
import torch
import torch.nn as nn
from utils.half_cheetah  import HalfCheetah
from utils.sac_dataset   import SACDataset
from utils.utils         import live_plot, show_result, save_model, load_model

## Parameters
TRAIN           = False
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
TARGET_UPDATE_TAU= 0.005
EPISODES        = 2000   # total episode
TARGET_ENTROPY  = -6.0
ALPHA           = 0.01
LEARNING_RATE_ALPHA= 0.01
# Memory
MEMORY_CAPACITY = 100000
BATCH_SIZE = 256
EPOCH_SIZE = 2
# Other
visulaize_step = 5
MAX_STEP = 1024         # maximun available step per episode
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
SAVE_DIR = current_directory + "/model/tutorial_continuos_2_1_SAC"
FILE_NAME = "135_actor"

sim = HalfCheetah(None)
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

## Parameters
# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 17
N_OUTPUT    = sim.env.action_space.shape[0]      # 6

# Memory
memory = SACDataset(MEMORY_CAPACITY)

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
    
    def train(self, target, state, action, optimizer):
        criterion = torch.nn.SmoothL1Loss()
        optimizer.zero_grad()
        loss = criterion(self.forward(state, action) , target)
        loss.mean().backward()
        optimizer.step()

    def update(self, target_net:nn.Module):
        for target_param, param in zip(target_net.parameters(), self.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TARGET_UPDATE_TAU) + param.data * TARGET_UPDATE_TAU)

# Initialize network
actor_net = ActorNetwork(N_INPUTS, N_OUTPUT).to(device)
q1_net = QNetwork(N_INPUTS, N_OUTPUT).to(device)
q2_net = QNetwork(N_INPUTS, N_OUTPUT).to(device)
target_q1_net = QNetwork(N_INPUTS, N_OUTPUT).to(device)
target_q2_net = QNetwork(N_INPUTS, N_OUTPUT).to(device)
alpha = torch.tensor(np.log(ALPHA))
alpha.requires_grad = True

target_q1_net.load_state_dict(q1_net.state_dict())
target_q2_net.load_state_dict(q2_net.state_dict())

# Optimizer
actor_optimizer   = torch.optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE)
q1_optimizer = torch.optim.AdamW(q1_net.parameters(), lr=LEARNING_RATE)
q2_optimizer = torch.optim.AdamW(q2_net.parameters(), lr=LEARNING_RATE)
alpha_optimizer   = torch.optim.AdamW([alpha], lr=LEARNING_RATE_ALPHA)

def optimize_model(batch):

    s, a, r, next_s = batch

    # Calculate 
    with torch.no_grad():
        next_a, next_logprob = actor_net(next_s)
        next_entropy = -alpha.exp() * next_logprob.mean(dim=1)

        next_q1 = target_q1_net(next_s, next_a)
        next_q2 = target_q2_net(next_s, next_a)

        next_min_q = torch.min(torch.cat([next_q1, next_q2], dim=1), 1)[0]
        target = r + DISCOUNT_FACTOR * (next_min_q + next_entropy).unsqueeze(1)

    q1_net.train(target, s, a, q1_optimizer)
    q2_net.train(target, s, a, q2_optimizer)

    action, logprob_batch = actor_net(s)
    entropy = -alpha.exp() * logprob_batch.mean(dim=1)
    q1 = q1_net(s, action)
    q2 = q2_net(s, action)
    
    min_q = torch.min(torch.cat([q1, q2],dim=1), 1)[0]
    actor_loss = -entropy - min_q
    actor_optimizer.zero_grad()
    actor_loss.mean().backward()
    actor_optimizer.step()

    alpha_optimizer.zero_grad()
    alpha_loss = - alpha.exp() * (logprob_batch.detach() + TARGET_ENTROPY).mean()
    alpha_loss.backward()
    alpha_optimizer.step()

    q1_net.update(target_q1_net)
    q2_net.update(target_q2_net)

total_steps = []
step_done_set = []
if TRAIN:
    for episode in range(1, EPISODES + 1):

        # 0. Reset environment
        state_curr, _ = sim.env.reset()
        state_curr = torch.tensor(state_curr, dtype=torch.float32, device=device)

        # Running one episode
        total_reward = 0.0
        for step in range(MAX_STEP):
            # 1. Get action from policy network
            action, logprob = actor_net(state_curr)

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done = sim.step(action.detach().cpu().numpy())
            total_reward += reward

            # 3. Update state
            state_next = torch.tensor(state_next, dtype=torch.float32, device=device)

            # 4. Save data
            memory.push(
                state_curr.unsqueeze(0),
                action.unsqueeze(0),
                torch.tensor([reward], device=device).unsqueeze(0),
                state_next.unsqueeze(0),
            )

            # 5. Update state
            state_curr = state_next

            # 6. Learning
            if (len(memory) > BATCH_SIZE):
                for _ in range(EPOCH_SIZE):
                    optimize_model(memory.sample(BATCH_SIZE))

            if done:
                break

        ## Episode is finished
        print(episode, "\t", total_reward)
        
        # Save episode reward
        step_done_set.append(total_reward)
        # Visualize
        if episode % visulaize_step == 0:
            if (len(total_steps) != 0) and (np.mean(step_done_set) >= max(total_steps)):
                save_model(actor_net, SAVE_DIR, "actor", episode)
            total_steps.append(np.mean(step_done_set))
            print("#{}: ".format(episode), np.mean(step_done_set).astype(int))
            live_plot(total_steps, visulaize_step)
            step_done_set = []

    # Turn the sim off
    sim.env.close()

    # Show the results
    show_result(total_steps, visulaize_step, SAVE_DIR)

else:
    sim = HalfCheetah()
    actor_net = load_model(actor_net, SAVE_DIR, FILE_NAME)

    # 0. Reset environment
    state_curr, _ = sim.env.reset()
    state_curr = torch.tensor(state_curr, dtype=torch.float32, device=device)

    # Running one episode
    for step in range(MAX_STEP):
        # 1. Get action from policy network
        action, logprob = actor_net(state_curr)

        # 2. Run simulation 1 step (Execute action and observe reward)
        state_next, reward, done = sim.step(action.detach().cpu().numpy())
        state_curr = torch.tensor(state_next, dtype=torch.float32, device=device)

# Turn the sim off
sim.env.close()
