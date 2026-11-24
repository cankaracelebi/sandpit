"""Actor-critic network for pixel + vector observations with MultiDiscrete actions."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class NatureCNN(nn.Module):
    """Atari-style encoder; input uint8 (B, C, H, W) with H=W in {48, 64, 84}."""

    def __init__(self, in_ch: int, size: int = 64, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_ch, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, in_ch, size, size)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n, out_dim)), nn.ReLU())

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x_uint8.float() / 255.0))


class ActorCritic(nn.Module):
    def __init__(self, nvec: tuple[int, ...], vec_dim: int, frame_stack: int = 4, size: int = 64, hidden: int = 512):
        super().__init__()
        self.nvec = tuple(int(n) for n in nvec)
        self.cnn = NatureCNN(3 * frame_stack, size, hidden)
        self.vec = nn.Sequential(layer_init(nn.Linear(vec_dim, 64)), nn.Tanh())
        self.trunk = nn.Sequential(layer_init(nn.Linear(hidden + 64, hidden)), nn.ReLU())
        self.actor = layer_init(nn.Linear(hidden, sum(self.nvec)), std=0.01)
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def features(self, pov: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([self.cnn(pov), self.vec(vec)], dim=1))

    def value(self, pov: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self.critic(self.features(pov, vec)).squeeze(-1)

    def act(self, pov: torch.Tensor, vec: torch.Tensor, action: torch.Tensor | None = None, deterministic: bool = False):
        h = self.features(pov, vec)
        logits = torch.split(self.actor(h), self.nvec, dim=1)
        dists = [Categorical(logits=l) for l in logits]
        if action is None:
            action = torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in dists], dim=1)
        logp = torch.stack([d.log_prob(a) for d, a in zip(dists, action.unbind(1))], dim=1).sum(1)
        ent = torch.stack([d.entropy() for d in dists], dim=1).sum(1)
        return action, logp, ent, self.critic(h).squeeze(-1)
