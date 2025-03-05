import torch
import torch.nn as nn
import numpy as np

############## transformer PARAMETERS ##############
nhead = 4
dff = 1024 
num_layers = 4
dropout = 0.3
#################################################### 


class CNN_transformer_simple(nn.Module):
    def __init__(self, nhead=nhead, dff=dff, num_layers=num_layers):
        super(CNN_transformer_simple, self).__init__()
        self.cnn1 = nn.Conv1d(4, 32, 8, padding=0)
        self.cnn2 = nn.Conv1d(32, 64, 8, padding=0)
        self.cnn3 = nn.Conv1d(64, 64, 7, padding=0) 
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=nhead,
            dim_feedforward=dff, 
            dropout=dropout,
            batch_first=True #(batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear1 = nn.Linear(57*64, 32)  
        self.linear2  = nn.Linear(32, 1)
    def forward(self, x):
        x = self.maxpool(torch.relu(self.cnn1(x))) # 501 -> 247
        x = self.maxpool(torch.relu(self.cnn2(x))) # -> 120  
        x = self.maxpool(torch.relu(self.cnn3(x))) # -> 57
        x = self.transformer(x.permute(0,2,1))
        x = torch.relu(self.linear1(x.flatten(start_dim=1)))
        out = torch.sigmoid(self.linear2(x))
        return out
    def summary(self):
        # Print model architecture and number of parameters
        print("\nModel Architecture:")
        print(self)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {num_params}")