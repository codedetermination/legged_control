# policy_network.py
import torch
import torch.nn as nn
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, in_dim: int = 450):
        super().__init__()
        self.module_list = nn.ModuleList([
            nn.Linear(in_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.module_list:
            x = m(x)
        return x

class Policy(nn.Module):
    def __init__(self, state_dim: int = 450, action_dim: int = 12,
                 log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__()
        self.model = MLP(state_dim)
        self.mean_decoder = nn.Sequential(
            nn.ELU(),
            nn.Linear(64, action_dim),
        )
        self.std_decoder = nn.Linear(64, action_dim)

        self.last_activation = nn.Tanh()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.model(state)
        mean = self.last_activation(self.mean_decoder(x))
        log_std = self.std_decoder(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, log_std, std

