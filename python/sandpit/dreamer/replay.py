"""Sequence replay for Dreamer: one stream per agent, sampled as fixed-length chunks.

Step convention (DreamerV3): row t holds obs_t, the reward received on arriving at obs_t, flags
is_first / is_terminal, and the action taken *at* obs_t. With the env's same-step auto-reset,
the terminal frame is never rendered, so a terminal row repeats the last frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class StreamReplay:
    def __init__(self, n_streams: int, capacity_per_stream: int, pov_shape: tuple, vec_dim: int, n_act: int, seed: int = 0):
        self.n, self.cap = n_streams, capacity_per_stream
        H, W, C = pov_shape
        self.pov = np.zeros((n_streams, capacity_per_stream, H, W, C), np.uint8)
        self.vec = np.zeros((n_streams, capacity_per_stream, vec_dim), np.float32)
        self.act = np.zeros((n_streams, capacity_per_stream, n_act), np.int64)
        self.rew = np.zeros((n_streams, capacity_per_stream), np.float32)
        self.first = np.zeros((n_streams, capacity_per_stream), bool)
        self.term = np.zeros((n_streams, capacity_per_stream), bool)
        self.ptr = np.zeros(n_streams, np.int64)     # total rows written per stream
        self.rng = np.random.default_rng(seed)

    def add(self, i: int, pov, vec, rew: float, first: bool, term: bool, act=None) -> None:
        k = self.ptr[i] % self.cap
        self.pov[i, k], self.vec[i, k], self.rew[i, k] = pov, vec, rew
        self.first[i, k], self.term[i, k] = first, term
        if act is not None:
            self.act[i, k] = act
        self.ptr[i] += 1

    def set_last_action(self, i: int, act) -> None:
        self.act[i, (self.ptr[i] - 1) % self.cap] = act

    @property
    def size(self) -> int:
        return int(np.minimum(self.ptr, self.cap).sum())

    def sample(self, batch: int, length: int) -> dict:
        filled = np.minimum(self.ptr, self.cap)
        ok = np.where(filled > length + 1)[0]
        streams = self.rng.choice(ok, size=batch)
        out = {k: [] for k in ("pov", "vec", "act", "rew", "first", "term")}
        for s in streams:
            n = filled[s]
            # valid (non-wrapped) start in chronological order
            oldest = self.ptr[s] - n
            start = oldest + self.rng.integers(0, n - length)
            idx = np.arange(start, start + length) % self.cap
            for k in out:
                out[k].append(getattr(self, k)[s, idx])
        b = {k: np.stack(v) for k, v in out.items()}
        b["first"][:, 0] = True  # every chunk starts from a fresh state
        return b

    def save(self, path: str | Path) -> None:
        filled = np.minimum(self.ptr, self.cap)
        np.savez_compressed(path, pov=self.pov, vec=self.vec, act=self.act, rew=self.rew, first=self.first,
                            term=self.term, ptr=self.ptr, filled=filled)

    @classmethod
    def load(cls, path: str | Path, seed: int = 0) -> "StreamReplay":
        d = np.load(path)
        n, cap = d["pov"].shape[:2]
        r = cls(n, cap, d["pov"].shape[2:], d["vec"].shape[-1], d["act"].shape[-1], seed)
        for k in ("pov", "vec", "act", "rew", "first", "term", "ptr"):
            setattr(r, k, d[k])
        return r


class StreamWriter:
    """Turns vec-env transitions into replay rows (handles same-step auto-reset)."""

    def __init__(self, replay: StreamReplay, agent_ids: np.ndarray):
        self.r, self.ids = replay, agent_ids
        self.last_pov = None
        self.last_vec = None

    def start(self, pov: np.ndarray, vec: np.ndarray) -> None:
        for j, i in enumerate(self.ids):
            self.r.add(j, pov[i], vec[i], 0.0, True, False)
        self.last_pov, self.last_vec = pov[self.ids].copy(), vec[self.ids].copy()

    def step(self, actions: np.ndarray, pov: np.ndarray, vec: np.ndarray, rew: np.ndarray, done: np.ndarray,
             trunc: np.ndarray) -> None:
        for j, i in enumerate(self.ids):
            self.r.set_last_action(j, actions[i])
            if done[i]:
                # terminal row: repeat last frame, carry the final reward
                self.r.add(j, self.last_pov[j], self.last_vec[j], rew[i], False, not trunc[i], act=np.zeros_like(actions[i]))
                self.r.add(j, pov[i], vec[i], 0.0, True, False)
            else:
                self.r.add(j, pov[i], vec[i], rew[i], False, False)
        self.last_pov, self.last_vec = pov[self.ids].copy(), vec[self.ids].copy()
