import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot(total_steps, step):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(np.linspace(1, len(total_steps) + 1, len(total_steps), endpoint=False) * step, total_steps, c='red')
    plt.grid(True)

def live_plot(total_steps, step):
    plot(total_steps, step)
    plt.pause(0.001)

def show_result(total_steps, step):
    print('step mean:', np.mean(total_steps))
    print('step  std:', np.std(total_steps))
    print('step  min:', np.min(total_steps))
    print('step  max:', np.max(total_steps))
    plot(total_steps, step)
    plt.show()