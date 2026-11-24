"""Vectorised multi-process, multi-agent environment over several Godot instances.

All agents from all processes are flattened into one batch (process-major order). Each agent
carries metadata (policy name, team, global arena id) so trainers can split the batch into
policy groups (learners vs frozen opponents, hiders vs seekers, ...).

Auto-reset is "same-step": when `done[i]` is True the returned observation for agent i is
already the first observation of its next episode (the terminal frame is not rendered).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bridge import NVEC, GodotProcess, StepResult


@dataclass
class AgentMeta:
    policy: str
    team: int
    arena: int      # global arena id
    proc: int
    local: int      # index inside its process


@dataclass
class VecObs:
    pov: np.ndarray     # (N, H, W, 3) uint8
    vec: np.ndarray     # (N, V) float32
    reward: np.ndarray  # (N,) float32
    done: np.ndarray    # (N,) bool
    trunc: np.ndarray   # (N,) bool
    infos: list = field(default_factory=list)  # dicts with global "agent" index


class GodotVecEnv:
    def __init__(
        self,
        scenario: str,
        n_procs: int = 2,
        arenas_per_proc: int = 4,
        seed: int = 0,
        action_repeat: int = 4,
        pov: int = 64,
        shadows: bool = False,
        params: dict | None = None,
        mode: str = "train",
        window: tuple[int, int] = (256, 256),
        log_dir: str | os.PathLike | None = None,
        godot_args: tuple[str, ...] = (),
        tile_windows: bool = True,
        minimize: bool = True,
        user_args: dict | None = None,
    ):
        self.scenario = scenario
        self.procs: list[GodotProcess] = []
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        try:
            for p in range(n_procs):
                pos = (20 + (p % 5) * (window[0] + 8), 60 + (p // 5) * (window[1] + 40)) if tile_windows else None
                self.procs.append(
                    GodotProcess(
                        scenario,
                        mode=mode,
                        seed=seed * 1000 + p,
                        arenas=arenas_per_proc,
                        action_repeat=action_repeat,
                        pov=pov,
                        shadows=shadows,
                        params=params,
                        window=window,
                        position=pos,
                        godot_args=godot_args,
                        user_args={"minimize": int(minimize), **(user_args or {})},
                        log_path=(Path(log_dir) / f"godot_{scenario}_{p}.log") if log_dir else None,
                    )
                )
        except Exception:
            self.close()
            raise
        p0 = self.procs[0]
        self.pov_shape = p0.pov_shape
        self.vec_dim = p0.vec_dim
        self.nvec = tuple(p0.nvec)
        self.info = p0.info
        self.meta: list[AgentMeta] = []
        self.offsets = []
        for pi, pr in enumerate(self.procs):
            self.offsets.append(len(self.meta))
            for li, a in enumerate(pr.agents):
                self.meta.append(AgentMeta(a["policy"], int(a["team"]), pi * arenas_per_proc + int(a["arena"]), pi, li))
        self.num_agents = len(self.meta)
        self.policies = sorted({m.policy for m in self.meta})

    def indices(self, policy: str) -> np.ndarray:
        return np.array([i for i, m in enumerate(self.meta) if m.policy == policy], dtype=np.int64)

    def _gather(self, results: list[StepResult]) -> VecObs:
        infos = []
        for pi, r in enumerate(results):
            for d in r.infos:
                d = dict(d)
                d["agent"] = self.offsets[pi] + int(d["agent"])
                infos.append(d)
        return VecObs(
            pov=np.concatenate([r.pov for r in results]),
            vec=np.concatenate([r.vec for r in results]),
            reward=np.concatenate([r.reward for r in results]),
            done=np.concatenate([r.done for r in results]),
            trunc=np.concatenate([r.trunc for r in results]),
            infos=infos,
        )

    def reset(self, seed: int | None = None) -> VecObs:
        for i, p in enumerate(self.procs):
            p.send_reset(None if seed is None else seed * 1000 + i)
        return self._gather([p.recv() for p in self.procs])

    def step(self, actions: np.ndarray) -> VecObs:
        actions = np.asarray(actions).reshape(self.num_agents, len(NVEC))
        for p, off in zip(self.procs, self.offsets):
            p.send_step(actions[off : off + p.n])
        return self._gather([p.recv() for p in self.procs])

    def set_params(self, params: dict, arenas: list[int] | None = None) -> None:
        """Update scenario params (e.g. curriculum stage) in all processes (or global arena ids)."""
        per = self.procs[0].info["n_arenas"]
        for pi, p in enumerate(self.procs):
            local = None if arenas is None else [a - pi * per for a in arenas if pi * per <= a < (pi + 1) * per]
            if local is not None and not local:
                continue
            p.call("set_params", params, local)

    def random_actions(self, rng: np.random.Generator) -> np.ndarray:
        return np.stack([rng.integers(0, n, size=self.num_agents) for n in self.nvec], axis=1).astype(np.int32)

    def close(self) -> None:
        for p in self.procs:
            p.close()
        self.procs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
