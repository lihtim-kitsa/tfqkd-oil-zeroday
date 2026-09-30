import torch
import torch.nn as nn

class FederatedMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=3):
        super(FederatedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
