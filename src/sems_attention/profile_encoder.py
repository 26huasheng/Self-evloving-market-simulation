from torch import nn


class ProfileEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(776, 128), nn.GELU(), nn.Linear(128, 64), nn.LayerNorm(64)
        )

    def forward(self, x):
        return self.net(x)
