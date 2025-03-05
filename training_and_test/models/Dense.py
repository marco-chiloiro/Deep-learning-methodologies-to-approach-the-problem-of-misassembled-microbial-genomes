import torch.nn as nn
import torch

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()
        self.linear1 = nn.Linear(501*4, 32) 
        self.linear2 = nn.Linear(32, 1) 
    def forward(self, x):
        x = torch.relu(self.linear1(x.flatten(start_dim=1))) 
        out = torch.sigmoid(self.linear2(x))
        return out
    def summary(self):
        # Print model architecture and number of parameters
        print("\nModel Architecture:")
        print(self)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {num_params}")