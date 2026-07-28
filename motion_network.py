"""PyTorch motion-estimation network."""

import torch.nn as nn


class MotionMLP(nn.Module):
    def __init__(self, window=15, hidden=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(window * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 9),
        )

    def forward(self, value):
        return self.network(value)
