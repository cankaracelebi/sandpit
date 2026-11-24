"""Load trained PPO checkpoints and run them on a GodotVecEnv (frame stacking handled here)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models import ActorCritic
from .ppo import pick_device


class PPOPolicy:
    def __init__(self, ckpt: str | Path, device: str = "auto"):
        d = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.cfg = d["cfg"]
        self.device = pick_device(device)
        self.K = int(self.cfg["frame_stack"])
        H = int(d["pov_shape"][0])
        self.nets = {}
        for p, sd in d["policies"].items():
            net = ActorCritic(tuple(d["nvec"]), int(d["vec_dim"]), self.K, H)
            net.load_state_dict(sd)
            self.nets[p] = net.to(self.device).eval()
        self.stack = None

    def reset(self, pov: np.ndarray) -> None:
        self.stack = np.repeat(pov[:, None], self.K, axis=1)

    def observe(self, pov: np.ndarray, done: np.ndarray) -> None:
        self.stack[:, :-1] = self.stack[:, 1:].copy()
        self.stack[:, -1] = pov
        if done.any():
            self.stack[done] = pov[done][:, None]

    @torch.no_grad()
    def act(self, vec: np.ndarray, policies: list[str], deterministic: bool = False) -> np.ndarray:
        n = self.stack.shape[0]
        x = self.stack.transpose(0, 1, 4, 2, 3).reshape(n, self.K * 3, *self.stack.shape[2:4])
        pov_t = torch.from_numpy(x).to(self.device)
        vec_t = torch.from_numpy(vec).to(self.device)
        out = np.zeros((n, 6), np.int32)
        pol = np.array(policies)
        for p, net in self.nets.items():
            idx = np.where(pol == p)[0]
            if len(idx):
                a, _, _, _ = net.act(pov_t[idx], vec_t[idx], deterministic=deterministic)
                out[idx] = a.cpu().numpy()
        return out
