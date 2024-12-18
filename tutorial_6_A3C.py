'''
value, policy based A3C (Asynchronous Advantage Actor Critic)
 - policy probability is parameterized as neural network
 - use multithreading to increase learning speed
 - policy model is updated for every n steps.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.multiprocessing as mp
from utils.cartpole      import BasicCartpole
from utils.policy_model  import Network
from utils.utils         import live_plot, show_result
from collections         import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'logprob', 'next_state', 'reward', 'done'))

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
N_INPUTS    = 4
N_OUTPUT    = 2
# Learning Parameters
LEARNING_RATE   = 0.0005 # optimizer
DISCOUNT_FACTOR = 0.99   # gamma
EPISODES        = 20000    # total episode
# Memory
MEMORY_SIZE = 64
# Other
visulaize_step = 25
MAX_STEP = 1000         # maximun available step per episode

def optimize_model(actor_net:ActorNetwork, critic_net:CriticNetwork, \
                   global_actor_net:ActorNetwork, global_critic_net:CriticNetwork, \
                   actor_optimizer, critic_optimizer, \
                   batch):

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

    for global_actor_param, local_actor_param in zip(global_actor_net.parameters(), actor_net.parameters()):
        global_actor_param._grad = local_actor_param.grad

    actor_optimizer.step()


    criterion = torch.nn.MSELoss()
    critic_loss = criterion(Vw_expected.detach(), Vw_curr)
    critic_loss.backward()  

    for global_critic_param, local_critic_param in zip(global_critic_net.parameters(), critic_net.parameters()):
        global_critic_param._grad = local_critic_param.grad

    critic_optimizer.step()

    actor_net.load_state_dict(global_actor_net.state_dict())
    critic_net.load_state_dict(global_critic_net.state_dict())


def run(global_actor_net:ActorNetwork, global_critic_net:CriticNetwork, rank:int):

    sim = BasicCartpole(None)

    # Memory
    memory = []

    # Initialize network
    actor_net = ActorNetwork(N_INPUTS, N_OUTPUT).to(device)
    critic_net = CriticNetwork(N_INPUTS, N_OUTPUT).to(device)

    actor_net.load_state_dict(global_actor_net.state_dict())
    critic_net.load_state_dict(global_critic_net.state_dict())

    # Optimizer
    actor_optimizer = torch.optim.AdamW(global_actor_net.parameters(), lr=LEARNING_RATE)
    critic_optimizer = torch.optim.AdamW(global_critic_net.parameters(), lr=LEARNING_RATE)

    for episode in range(1, EPISODES + 1):

        # 0. Reset environment
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
                optimize_model(actor_net,
                               critic_net,
                               global_actor_net,
                               global_critic_net,
                               actor_optimizer,
                               critic_optimizer,
                               Transition(*zip(*memory)))
                memory = []

            # 6. Update state
            state_curr = state_next

            if done:
                break
        ## Episode is finished

    # Turn the sim off
    sim.env.close()

def test(global_actor_net:ActorNetwork, global_critic_net:CriticNetwork, rank:int):

    sim = BasicCartpole(None)

    # Initialize network
    test_actor_net = ActorNetwork(N_INPUTS, N_OUTPUT).to(device)

    test_actor_net.load_state_dict(global_actor_net.state_dict())

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
            prob = test_actor_net(state_curr)
            action = Categorical(prob).sample()

            # 2. Run simulation 1 step (Execute action and observe reward)
            state_next, reward, done, _, _ = sim.env.step(action.item())

            # 3. Update state
            state_next = torch.tensor(state_next, dtype=torch.float32, device=device)

            # 5. Update step of current episode
            step_done += 1

            # 6. Update state
            state_curr = state_next

            if done:
                break
        ## Episode is finished
        test_actor_net.load_state_dict(global_actor_net.state_dict())
        
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

if __name__ == '__main__':
    mp.set_start_method('spawn')

    global_actor_net = ActorNetwork(N_INPUTS, N_OUTPUT).to(device)
    global_critic_net = CriticNetwork(N_INPUTS, N_OUTPUT).to(device)

    global_actor_net.share_memory()
    global_critic_net.share_memory()

    processes = []
    for rank in range(5):
        if rank == 0:
            p = mp.Process(target=test, args=(global_actor_net, global_critic_net, rank,))
        else:
            p = mp.Process(target=run, args=(global_actor_net, global_critic_net, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()