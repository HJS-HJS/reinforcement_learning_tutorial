'''
value, policy based A2C (Advantage Actor Critic)
 - policy probability is parameterized as neural network
 - use A = Gt - Vt = r + gamma * Vt+1 - Vt to reduce variance
 - policy model is updated for every n steps.
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
                        ('state', 'logprob', 'next_state', 'reward', 'done'))

class ActorNetwork(Network):
    def __init__(self, n_state:int = 4, n_action:int = 2):
        super(ActorNetwork, self).__init__(n_state, n_action)
        self.layer.add_module("linear", nn.Linear(128, n_action))
        self.layer.add_module("softmax", nn.Softmax(dim=0))

class CriticNetwork(Network):
    def __init__(self, n_state:int = 4, n_action:int = 2):
        super(CriticNetwork, self).__init__(n_state, n_action)
        self.layer.add_module("linear", nn.Linear(128, 1))

## Parameters
# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 4
N_OUTPUT    = sim.env.action_space.n             # 2
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
EPISODES        = 20000    # total episode
# Memory
MEMORY_SIZE = 64
# Other
visulaize_step = 25
MAX_STEP = 1000         # maximun available step per episode

# Memory
memory = []

# Initialize network
actor_net = ActorNetwork(N_INPUTS, N_OUTPUT).to(device)
critic_net = CriticNetwork(N_INPUTS, N_OUTPUT).to(device)

# Optimizer
actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
critic_optimizer = torch.optim.AdamW(critic_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

def optimize_model(batch):

    state_batch     = torch.cat(batch.state)
    logprob_batch   = torch.cat(batch.logprob)
    next_state_batch= torch.cat(batch.next_state)
    reward_batch    = torch.cat(batch.reward)
    done_batch      = torch.cat(batch.done)

    Vw_curr = critic_net(state_batch)
    Vw_next = critic_net(next_state_batch) * done_batch
    Vw_expected = reward_batch + DISCOUNT_FACTOR * Vw_next
    TD_error = Vw_expected - Vw_curr

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    
    actor_loss = (-logprob_batch * TD_error.detach().squeeze(1)).mean()
    actor_loss.backward()
    actor_optimizer.step()

    criterion = torch.nn.MSELoss()
    critic_loss = criterion(Vw_expected.detach(), Vw_curr)
    critic_loss.backward()
    critic_optimizer.step()


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
        prob = actor_net(state_curr)
        action = Categorical(prob).sample()

        # 2. Run simulation 1 step (Execute action and observe reward)
        state_next, reward, done, _, _ = sim.env.step(action.item())

        # 3. Update state
        state_next = torch.tensor(state_next, dtype=torch.float32, device=device)

        # 4. Save data
        memory.append(Transition(
            state_curr.unsqueeze(0),
            torch.log(prob[action]).unsqueeze(0),
            state_next.unsqueeze(0),
            torch.tensor([reward], device=device).unsqueeze(0),
            torch.tensor([not done], device=device).unsqueeze(0)
        ))

        # 5. Learning
        if len(memory) % MEMORY_SIZE == 0:
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
