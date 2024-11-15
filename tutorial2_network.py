from utils.cartpole import BasicCartpole
import numpy as np
import tensorflow as tf


sim = BasicCartpole()
sim.render_scene()


# Policy
# Network Param
N_INPUTS = 4  # == sim.env.observation_space.shape[0]
N_HIDDEN = 16  # CartPole은 간단한 환경이므로 16개의 뉴런을 사용
N_OUTPUTS = 1  # 왼쪽(0)으로 이동할 확률을 출력
initializer = tf.keras.initializers.he_normal()

# Network Set
model = tf.keras.Sequential([tf.keras.Input(shape=(N_INPUTS,)),
                             tf.keras.layers.Dense(units=N_HIDDEN, activation=tf.nn.elu, kernel_initializer=initializer),
                             tf.keras.layers.Dense(units=N_OUTPUTS, activation=tf.nn.sigmoid, kernel_initializer=initializer)])

# Network Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='mse')

model.summary()

n_max_steps = 1000
frames, totals = [], []

for episode in range(50):
    if episode % 5 == 0:
        print('episode : {}'.format(episode))
    episode_rewards = 0
    obs, _ = sim.env.reset()
    for step in range(n_max_steps):
        sim.render_scene()
        # labet setting
        label = np.ones(1) if obs[2] > 0 else np.zeros(1)
        # train
        model.fit(obs.reshape(-1,4), label, epochs=1)
        # apply results 
        # convert prediction result 0 or 1
        ans = 0 if model.predict(obs.reshape(-1,4)) < 0.5 else 1
        # 1 step simulation
        obs, reward, done, info, _ = sim.env.step(ans)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
sim.env.close()

print('totals mean :', np.mean(totals))
print('totals std :', np.std(totals))
print('totals min :', np.min(totals))
print('totals max :', np.max(totals))