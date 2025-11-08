import torch.nn as nn
import torch.nn.functional as F


class ClassicRegressionMLP(nn.Module):
    NAME = "ClassicRegressionMLP"

    def __init__(self, input_features: int = 8, hidden_units: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


class DropoutRegressionMLP(nn.Module):
    NAME = "DropoutRegressionMLP"

    def __init__(self, input_features: int = 8, hidden_units: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


class NormalizedRegressionMLP(nn.Module):
    NAME = "NormalizedRegressionMLP"

    def __init__(self, input_features: int = 8, hidden_units: int = 64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(),
            nn.Linear(hidden_units // 2, 1),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualRegressionMLP(nn.Module):
    NAME = "ResidualRegressionMLP"

    def __init__(self, input_features: int = 8, hidden_units: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.bn2 = nn.BatchNorm1d(hidden_units // 2)
        self.fc3 = nn.Linear(hidden_units // 2, 1)
        self.use_skip = input_features == (hidden_units // 2)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        residual = out

        out = F.relu(self.bn2(self.fc2(out)))
        if self.use_skip and residual.shape == out.shape:
            out = out + residual
        out = self.fc3(out)

        return out
