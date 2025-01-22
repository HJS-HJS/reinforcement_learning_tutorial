import numpy as np
import torch
from utils.cartpole    import BasicCartpole
from utils.dqn_model   import DQNNetwork
from utils.dqn_dataset import DQNDataset
from utils.utils       import live_plot, show_result

sim = BasicCartpole(None)
device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

## Parameters
# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 4
N_OUTPUT    = sim.env.action_space.n             # 2
BATCH_SIZE  = 32
MEMORY_CAPACITY = 10000
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
EPISODES        = 500    # total episode
UPDATE_STEPS    = 500    # policy update step (copy behavior network to target network)
# Greedy parameter
EPSILON_START = 0.9
EPSILON_END   = 0.05
EPSILON_DECAY = 1000
# Other
visulaize_step = 25
MAX_STEP = 1000         # maximun available step per episode

# Initialize network
target_net = DQNNetwork(N_INPUTS, N_OUTPUT).to(device)
policy_net = DQNNetwork(N_INPUTS, N_OUTPUT).to(device)

# Optimizer
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

# Apply policy_net parameter to target_net
target_net.load_state_dict(policy_net.state_dict())

# Generate dataset
replay_memory = DQNDataset(MEMORY_CAPACITY)

steps_done = 0
def get_action(state, policy_net):
    # epsilon greedy
    global steps_done
    sample = np.random.uniform(0, 1, 1)
    eps_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * steps_done / EPSILON_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[np.random.randint(2)]], device=device, dtype=torch.long)

def optimize_model(policy_net, target_net):
    if len(replay_memory) < BATCH_SIZE:
        return
    # Sample minibatch of trasnisions(s, a, r, s) from replay buffer
    batch = replay_memory.sample(BATCH_SIZE)

    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    ## Q(s_t,a_t)
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    ## y = r + gamma * maxQ(s_t, a_t)   (from target network)
    # Find terminated step (when next state is not exist)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # print(non_final_next_states)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute TD target
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Compute Huber loss
    criterion = torch.nn.SmoothL1Loss()
    # gradient step of (y - Q(s_t,a_t))^2 -> (Q(s_t,a_t) - y) delta Q
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


total_steps = []
step_done_set = []
for episode in range(1, EPISODES + 1):

    # 0. Reset environment
    step_done = 0
    state_curr, _ = sim.env.reset()
    state_curr = torch.tensor(state_curr, dtype=torch.float32, device=device).unsqueeze(0)

    for step in range(MAX_STEP):
        # 1. Get action from policy network
        action = get_action(state_curr, policy_net)

        # 2. Run simulation 1 step (Execute action and observe reward)
        state_next, reward, done, _, _ = sim.env.step(action.item())

        # 3. Update state
        state_next = None if done else torch.tensor(state_next, dtype=torch.float32, device=device).unsqueeze(0)

        # 4. Store transition to replay buffer D as tensor
        replay_memory.push(
            state_curr, 
            action,
            state_next,
            torch.tensor([reward], device=device))

        # 5. Learning
        optimize_model(policy_net, target_net)
        
        # 6. Update step of current episode
        step_done += 1

        # 7. Update state
        state_curr = state_next

        # 8. Update target_net as using policy_net
        if steps_done % UPDATE_STEPS == 0:
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

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
