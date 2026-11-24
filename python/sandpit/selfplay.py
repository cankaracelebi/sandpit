"""Self-play utilities: a snapshot pool of past policies and Elo ratings."""
from __future__ import annotations

import copy

import numpy as np


class Elo:
    def __init__(self, k: float = 16.0, init: float = 1000.0):
        self.k = k
        self.init = init
        self.r: dict[str, float] = {}

    def get(self, name: str) -> float:
        return self.r.setdefault(name, self.init)

    def update(self, a: str, b: str, score_a: float) -> None:
        """score_a in {1, 0.5, 0} (win / draw / loss for a)."""
        ra, rb = self.get(a), self.get(b)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        self.r[a] = ra + self.k * (score_a - ea)
        self.r[b] = rb + self.k * ((1.0 - score_a) - (1.0 - ea))


class OpponentPool:
    """Frozen snapshots of a learner. sample(): latest with prob `latest_prob`, else uniform older."""

    def __init__(self, max_size: int = 20, latest_prob: float = 0.8, seed: int = 0):
        self.max_size = max_size
        self.latest_prob = latest_prob
        self.snaps: list[tuple[str, dict]] = []
        self.rng = np.random.default_rng(seed)

    def add(self, name: str, state_dict: dict) -> None:
        sd = {k: v.detach().to("cpu").clone() for k, v in state_dict.items()}
        self.snaps.append((name, sd))
        if len(self.snaps) > self.max_size:
            # keep the very first snapshot as a fixed weak reference point
            self.snaps.pop(1)

    def sample(self) -> tuple[str, dict]:
        if len(self.snaps) == 1 or self.rng.random() < self.latest_prob:
            return self.snaps[-1]
        return self.snaps[int(self.rng.integers(0, len(self.snaps) - 1))]

    def __len__(self) -> int:
        return len(self.snaps)


def clone_module(m):
    return copy.deepcopy(m)
