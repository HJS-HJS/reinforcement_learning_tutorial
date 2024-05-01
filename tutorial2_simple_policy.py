from render_notebook import render_cart_pole

import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def plot_cart_pole(env, obs):
    render_cart_pole(env, obs)
    
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

import gym

env = gym.make("CartPole-v1", render_mode='human')
obs = env.reset()
render_cart_pole(env, obs)

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle <0 else 1

frames, totals = [], []
for episode in range(20):
    episode_rewards = 0
    obs, _temp = env.reset()
    for step in range(1000):  # 최대 스텝을 1000번으로 설정
        img = render_cart_pole(env, obs)
        
        action = basic_policy(obs)
        obs, reward, done, info, _temp = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))