import torch
import torch.nn as nn

class MLPDecoder(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        self.net[-1].bias.data = torch.tensor([1.0, 0.0])

    def forward(self, x):
        return self.net(x)
