'''
policy based reinforcement learning
 - policy probability is parameterized as neural network
 - use Gt
 - policy model is updated only when each episode is finished.
'''

import numpy as np
import torch
from torch.distributions    import Categorical
from utils.cartpole         import BasicCartpole
from utils.REINFORCE_model  import Network
from utils.utils            import live_plot, show_result

sim = BasicCartpole(None)
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

## Parameters
# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 4
N_OUTPUT    = sim.env.action_space.n             # 2
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
EPISODES        = 2000    # total episode
# Other
visulaize_step = 25
MAX_STEP = 1000         # maximun available step per episode

# Initialize network
target_net = Network(N_INPUTS, N_OUTPUT).to(device)

# Optimizer
optimizer = torch.optim.AdamW(target_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

def get_action(state, target_net):
    prob = target_net(state)
    action = Categorical(prob).sample()
    return prob[action], action.item()

def optimize_model(memory):
    
    Gt = 0
    optimizer.zero_grad()

    for r, prob in memory[::-1]:
        Gt = r + DISCOUNT_FACTOR * Gt
        loss = -torch.log(prob) * Gt
        loss.backward()

    optimizer.step()


total_steps = []
step_done_set = []
for episode in range(1, EPISODES + 1):

    # 0. Reset environment
    memory = []
    step_done = 0
    state_curr, _ = sim.env.reset()
    state_curr = torch.tensor(state_curr, dtype=torch.float32, device=device)

    # Running one episode
    for step in range(MAX_STEP):
        # 1. Get action from policy network
        prob = target_net(state_curr)
        action = Categorical(prob).sample()

        # 2. Run simulation 1 step (Execute action and observe reward)
        state_next, reward, done, _, _ = sim.env.step(action.item())

        # 3. Update state
        state_next = None if done else torch.tensor(state_next, dtype=torch.float32, device=device)

        # 4. Store data to memory
        memory.append([
            reward,
            prob[action]
            ])
        
        # 5. Update step of current episode
        step_done += 1

        # 6. Update state
        state_curr = state_next

        if done:
            break
    ## Episode is finished

    # 7. Learning
    optimize_model(memory)
    
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
