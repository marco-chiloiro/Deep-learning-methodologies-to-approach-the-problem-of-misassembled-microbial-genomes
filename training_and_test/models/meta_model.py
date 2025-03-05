import torch.nn as nn
import torch

class meta_model(nn.Module):
    # stack together 2 models
    def __init__(self, modA, modB):
        super(meta_model, self).__init__()
        self.modA = modA.eval()
        self.modB = modB.eval()
        self.classifier = nn.Linear(2, 1) 
    def forward(self, x):
        x1 = self.modA(x)
        x2 = self.modB(x)
        x = torch.cat((x1, x2), dim=1)
        out = torch.sigmoid(self.classifier(x))        
        return out
    def summary(self):
        # Print model architecture and number of parameters
        print("\nModel Architecture:")
        print(self)
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {num_params}")