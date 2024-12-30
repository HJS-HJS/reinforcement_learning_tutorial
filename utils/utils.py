import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot(total_steps, step):
    plt.figure(1)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(np.linspace(1, len(total_steps) + 1, len(total_steps), endpoint=False) * step, total_steps, c='red')
    plt.grid(True)

def live_plot(total_steps, step):
    plot(total_steps, step)
    plt.pause(0.0001)

def show_result(total_steps, step, save_dir:None):
    print('step mean:', np.mean(total_steps))
    print('step  std:', np.std(total_steps))
    print('step  min:', np.min(total_steps))
    print('step  max:', np.max(total_steps))
    plot(total_steps, step)
    if save_dir is not None:
        plt.savefig(save_dir + "/results.png")
    plt.show()

def save_model(network:torch.nn.Module, save_dir:str, name:str, episode:int):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{episode}_{name}.pth")
    torch.save(network.state_dict(), save_path)
    print(f"\tModels saved at episode {episode}")

def load_model(network:torch.nn.Module, save_dir:str, name:str):
    save_path = os.path.join(save_dir, f"{name}.pth")
    if os.path.exists(save_path):
        state_dict = torch.load(save_path, weights_only=True)
        network.load_state_dict(state_dict)
        print(f"\tModels loaded from {save_path}")
    else:
        print(f"\tNo saved model found {save_path}")
    return network