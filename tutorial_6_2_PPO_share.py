'''
PPO (Proximal Policy Optimization)
 - actor and critic share network parameters.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils.cartpole      import BasicCartpole
from utils.policy_model  import Network
from utils.utils         import live_plot, show_result
from collections import namedtuple

sim = BasicCartpole(None)
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

Transition = namedtuple('Transition',
                        ('state', 'action', 'prob', 'next_state', 'reward', 'done'))

class ACNetwork(Network):
    def __init__(self, n_state:int = 4, n_action:int = 2):
        super(ACNetwork, self).__init__(n_state, n_action)
        self.actor_layer = nn.Sequential(
            nn.Linear(128, n_action),
            nn.Softmax(dim=-1),
        )
        self.critic_layer = nn.Sequential(
            nn.Linear(128, 1),
        )

    def Actor(self, x: torch.tensor):
        x1 = self.layer(x)
        return self.actor_layer(x1)

    def Critic(self, x: torch.tensor):
        x1 = self.layer(x)
        return self.critic_layer(x1)

## Parameters
# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 4
N_OUTPUT    = sim.env.action_space.n             # 2
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
ADVANTAGE_LAMBDA = 0.95
EPISODES        = 20000    # total episode
# Memory
BATCH_SIZE = 64
EPOCH_SIZE = 5
CLIP_EPSILON = 0.05
# Other
visulaize_step = 25
MAX_STEP = 1000         # maximun available step per episode

# Memory
memory = []

# Initialize network
actor_critic_net = ACNetwork(N_INPUTS, N_OUTPUT).to(device)

# Optimizer
optimizer = torch.optim.AdamW(actor_critic_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

def optimize_model(batch):

    state_batch     = torch.cat(batch.state)
    action_batch  = torch.cat(batch.action)
    prob_old_batch  = torch.cat(batch.prob)
    next_state_batch= torch.cat(batch.next_state)
    reward_batch    = torch.cat(batch.reward)
    done_batch      = torch.cat(batch.done)

    for _ in range(EPOCH_SIZE):

        # Calculate the current probability
        prob = actor_critic_net.Actor(state_batch)
        prob_batch = prob[torch.arange(prob.size(0)),action_batch]
        # Calculate the importance ratio
        imp_ratio = prob_batch / prob_old_batch.detach()

        ## Calculate A from V
        # Calculate delta_t
        value_curr_set = actor_critic_net.Critic(state_batch)
        value_next_set = actor_critic_net.Critic(next_state_batch) * done_batch
        td_target = reward_batch + DISCOUNT_FACTOR * value_next_set
        delta_set = (td_target - value_curr_set).detach()
        # Calculate Advantage_set
        powers = (DISCOUNT_FACTOR * ADVANTAGE_LAMBDA) ** torch.arange(len(value_curr_set)).float()
        matrix = torch.tril(powers.view(-1, 1) / powers.view(1, -1)).to(device)
        advantage_set = torch.mul(delta_set.repeat(1, len(value_curr_set)), matrix)
        advantage_set = torch.sum(advantage_set, dim=0)

        # Clipping
        clip_advantage = torch.min(imp_ratio * advantage_set, torch.clamp(imp_ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantage_set)

        # Learning
        optimizer.zero_grad()

        # Gradient 
        criterion = torch.nn.MSELoss()
        loss = -clip_advantage.sum() + criterion(td_target.detach(), value_curr_set)

        loss.backward()
        optimizer.step()

total_steps = []
step_done_set = []
for episode in range(1, EPISODES + 1):

    # 0. Reset environment
    step_done = 0
    state_curr, _ = sim.env.reset()
    state_curr = torch.tensor(state_curr, dtype=torch.float32, device=device)

    # Running one episode
    for step in range(MAX_STEP):
        # 1. Get action from policy network
        prob = actor_critic_net.Actor(state_curr)
        action = Categorical(prob).sample()

        # 2. Run simulation 1 step (Execute action and observe reward)
        state_next, reward, done, _, _ = sim.env.step(action.item())

        # 3. Update state
        state_next = torch.tensor(state_next, dtype=torch.float32, device=device)
        # print(prob)
        # 4. Save data
        memory.append(Transition(
            state_curr.unsqueeze(0),
            action.unsqueeze(0),
            prob[action].unsqueeze(0),
            state_next.unsqueeze(0),
            torch.tensor([reward], device=device).unsqueeze(0),
            torch.tensor([not done], device=device).unsqueeze(0)
        ))

        # 5. Learning
        if (len(memory) % BATCH_SIZE == 0) or (done is True):
            optimize_model(Transition(*zip(*memory)))
            memory = []

        # 6. Update step of current episode
        step_done += 1

        # 7. Update state
        state_curr = state_next

        if done:
            break
    ## Episode is finished
    
    # Save episode reward
    step_done_set.append(step_done)
    # Visualize
    if episode % visulaize_step == 0:
        total_steps.append(np.mean(step_done_set))
        print("#{}: ".format(episode), np.mean(step_done_set).astype(int))
        live_plot(total_steps, visulaize_step)
        step_done_set = []

# Turn the sim off
sim.env.close()

# Show the results
show_result(total_steps, visulaize_step)
