import torch
import torch.nn as nn

class RegressionMLP(nn.Module):
    def __init__(self, input_features=8, hidden_units=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass"""
        return self.layers(x)