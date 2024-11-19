import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple, deque
import random

device = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available")
    device = torch.device('cuda')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQNDataset(Dataset):
    def __init__(self, n_que:int = 1000):
        """Dataset class for DQN.

        Args:
            n_que (int): size of que.
        """
        # Dataset
        self.data_list = deque([], maxlen=n_que)

    def push(self, *args):
        self.data_list.append(Transition(*args))

    def sample(self, batch_size):
        samples = random.sample(self.data_list, batch_size)
        return Transition(*zip(*samples))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == '__main__':
    dataset = DQNDataset()