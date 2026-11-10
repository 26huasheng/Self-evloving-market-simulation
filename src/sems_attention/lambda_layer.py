import torch
from torch import nn


class DynamicLambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.M0 = nn.Parameter(torch.eye(64))
        self.U = nn.Parameter(torch.randn(64, 8) * 0.02)
        self.V = nn.Parameter(torch.randn(8, 64) * 0.02)
        self.gate = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 8), nn.Sigmoid())

    def forward(self, e, p):
        return self.M0 @ p + self.U @ (self.gate(e) * (self.V @ p))

    def matrix(self, e):
        return self.M0 + (self.U * self.gate(e).unsqueeze(0)) @ self.V
