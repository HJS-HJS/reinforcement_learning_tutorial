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
obs, _ = env.reset()
render_cart_pole(env, obs)

n_inputs = 4  # == env.observation_space.shape[0]
n_hidden = 16  # CartPole은 간단한 환경이므로 16개의 뉴런을 사용
n_outputs = 1  # 왼쪽(0)으로 이동할 확률을 출력
initializer = tf.keras.initializers.he_normal()

model = tf.keras.Sequential([tf.keras.Input(shape=n_inputs),
                             tf.keras.layers.Dense(units=n_hidden, activation=tf.nn.elu, kernel_initializer=initializer),
                             tf.keras.layers.Dense(units=n_outputs, activation=tf.nn.sigmoid, kernel_initializer=initializer)])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='mse')

model.summary()

n_max_steps = 1000
frames, totals = [], []

for episode in range(50):
    if episode % 5 == 0:
        print('episode : {}'.format(episode))
    episode_rewards = 0
    obs, _ = env.reset()
    for step in range(n_max_steps):
        render_cart_pole(env, obs)
        label = np.ones(1) if obs[2] < 0 else np.zeros(1)
        model.fit(obs.reshape(-1,4), label, epochs=1)

        obs, reward, done, info, _ = env.step(model.predict(obs))
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
env.close()

print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))