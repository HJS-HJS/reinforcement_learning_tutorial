import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, n_state:int = 4, n_action:int = 2):
        
        super(Network, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        
    def forward(self, state):
        return self.layer(state)

if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device('cuda')

    model = Network().eval()
    print(list(model.children()))
