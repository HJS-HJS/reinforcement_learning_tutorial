from utils.cartpole import BasicCartpole
import numpy as np
from utils.utils import live_plot, show_result

sim = BasicCartpole()
sim.render_scene()

# policy
def basic_policy(obs):
    angle = obs[2]
    return 1 if angle > 0 else 0


total_steps = []
for episode in range(20):
    steps_done = 0
    obs, _temp = sim.env.reset()
    for step in range(1000):  # 최대 스텝을 1000번으로 설정
        sim.render_scene()
        
        action = basic_policy(obs)
        obs, reward, done, info, _temp = sim.env.step(action)
        steps_done += 1
        if done:
            break
    total_steps.append(steps_done)
    live_plot(total_steps, 1)

show_result(total_steps, 1)