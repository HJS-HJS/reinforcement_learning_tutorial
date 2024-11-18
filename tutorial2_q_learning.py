from utils.cartpole import BasicCartpole
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sim = BasicCartpole('None')

## Parameters
# Policy Parameters
N_INPUTS    = sim.env.observation_space.shape[0] # 4
N_OUTPUT    = sim.env.action_space.n             # 2
Observation = np.array([30, 30, 70, 70])
discrete    = np.array([0.25, 0.25, 0.01, 0.1])
# Learning Parameters
LEARNING_RATE   = 0.1   # alpha
DISCOUNT_FACTOR = 0.95  # gamma
EPISODES      = 60000
# Other
visulaize_step = 1000
MAX_STEP = 1000

# Gen policy table (q table)
q_table = np.random.uniform(
    low=0, 
    high=1, 
    size=(np.hstack([Observation, N_OUTPUT]))
    )

def get_discrete_state(state):
    discrete_state = state/discrete + Observation/2
    return tuple(discrete_state.astype(np.int))

def live_plot(total_steps):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(total_steps)
    # pause a bit so that plots are updated
    plt.pause(0.001)

total_steps = []
step_done_set = []
for episode in range(EPISODES):
    if episode%visulaize_step == 0:
        visulaize = True
    else: 
        visulaize = False

    step_done = 0
    observe, _ = sim.env.reset()
    state_curr = get_discrete_state(observe)

    for step in range(MAX_STEP):
        # 1. Get action
        action = np.argmax(q_table[state_curr])

        # 2. Run simulation 1 step
        observe, reward, done, _, _ = sim.env.step(action)
        
        # 3. Calculate reward
        step_done += 1
        
        if done:
            break
        # Update q_table
        else:
            # 4. update next state
            state_next = get_discrete_state(observe)

            # 5. Get next state q value
            maximum_future_q = np.max(q_table[state_next])

            # 6. Update step of current episode
            current_q = q_table[state_curr + (action,)]

            # 7. Calculate updated q value
            updated_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * maximum_future_q)

            # 8. Update current q value as updated q value
            q_table[state_curr + (action,)] = updated_q

            # 8. Update state
            state_curr = state_next

    ## Episode is finished
    # Save episode reward
    step_done_set.append(step_done)
    # Visualize
    if visulaize:
        total_steps.append(np.mean(step_done_set))
        print("#{}: ".format(episode), np.mean(step_done_set).astype(np.int))
        live_plot(total_steps, visulaize_step)
        step_done_set = []

# Turn the sim off
sim.env.close()

# Show the results
print('step mean:', np.mean(total_steps))
print('step  std:', np.std(total_steps))
print('step  min:', np.min(total_steps))
print('step  max:', np.max(total_steps))
# Show results with grapth
plt.figure(1)
plt.xlabel('Episode')
plt.ylabel('Duration')
plt.plot(total_steps)
plt.show()
