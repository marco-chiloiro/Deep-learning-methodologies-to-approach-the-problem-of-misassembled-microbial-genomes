import torch.nn as nn
import torch

class deep_CNN_simple(nn.Module):
    def __init__(self):
        super(deep_CNN_simple, self).__init__()
        self.cnn1 = nn.Conv1d(4, 16, 8, padding=0) 
        self.cnn2 = nn.Conv1d(16, 32, 8, padding=0) 
        self.cnn3 = nn.Conv1d(32, 64, 7, padding=0) 
        self.cnn4 = nn.Conv1d(64, 128, 7, padding=0)
        self.linear1 = nn.Linear(51*128, 32) 
        self.linear2 = nn.Linear(32, 1) 
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.maxpool(torch.relu(self.cnn1(x))) 
        x = self.maxpool(torch.relu(self.cnn2(x))) 
        x = self.maxpool(torch.relu(self.cnn3(x)))
        x = torch.relu(self.cnn4(x)) 
        x = torch.relu(self.linear1(x.flatten(start_dim=1))) 
        out = torch.sigmoid(self.linear2(x))
        return out
    def summary(self):
        # Print model architecture and number of parameters
        print("\nModel Architecture:")
        print(self)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {num_params}")