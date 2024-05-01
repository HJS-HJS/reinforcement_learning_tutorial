from cartpole import BasicCartpole
import numpy as np

sim = BasicCartpole()
sim.render_scene()

# policy
def basic_policy(obs):
    angle = obs[2]
    return 1 if angle > 0 else 0


totals = []
for episode in range(20):
    episode_rewards = 0
    obs, _temp = sim.env.reset()
    for step in range(1000):  # 최대 스텝을 1000번으로 설정
        sim.render_scene()
        
        action = basic_policy(obs)
        obs, reward, done, info, _temp = sim.env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))