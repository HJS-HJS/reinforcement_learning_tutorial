from render_notebook import render_cart_pole

import os, sys, time
import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 한글출력
# matplotlib.rc('font', family='AppleGothic')  # MacOS
matplotlib.rc('font', family='Malgun Gothic')  # Windows
# matplotlib.rc('font', family='NanumBarunGothic') # Linux
plt.rcParams['axes.unicode_minus'] = False

def plot_cart_pole(env, obs):
    render_cart_pole(env, obs)
    
def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

import gym

env = gym.make("CartPole-v1", render_mode='human')
obs = env.reset()
img = render_cart_pole(env, obs)

print('obs :', obs)
plot_cart_pole(env, obs)
print('env.action_space :', env.action_space)
time.sleep(5)