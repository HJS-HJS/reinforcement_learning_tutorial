'''
value, policy based reinforcement learning
 - policy probability is parameterized as neural network
 - use Gt
 - policy model is updated for every step.
 - actor and critic not share network parameters.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils.cartpole      import BasicCartpole
from utils.policy_model  import Network
from utils.utils         import live_plot, show_result

sim = BasicCartpole(None)
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

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
# Other
visulaize_step = 25
MAX_STEP = 1000         # maximun available step per episode

# Initialize network
actor_net = ActorNetwork(N_INPUTS, N_OUTPUT).to(device)
critic_net = CriticNetwork(N_INPUTS, N_OUTPUT).to(device)

# Optimizer
actor_optimizer = torch.optim.AdamW(actor_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
critic_optimizer = torch.optim.AdamW(critic_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

def optimize_model(s, p, a, r, s_next):
    
    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    Qw_curr = critic_net(s)
    Qw_next = torch.zeros(1, device=device) if s_next is None else critic_net(s_next)
    Qw_expected = r + DISCOUNT_FACTOR * Qw_next
    TD_error = Qw_expected - Qw_curr
    actor_loss = -torch.log(p) * TD_error.detach()
    actor_loss.backward()

    actor_optimizer.step()

    criterion = torch.nn.MSELoss()
    critic_loss = criterion((r + DISCOUNT_FACTOR * Qw_next).detach(), Qw_curr)
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
        state_next = None if done else torch.tensor(state_next, dtype=torch.float32, device=device)

        # 4. Learning
        optimize_model(
            s      = state_curr,
            p      = prob[action],
            a      = action,
            r      = reward,
            s_next = state_next
        )

        # 5. Update step of current episode
        step_done += 1

        # 6. Update state
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
