"""Win-rate driven curricula. `update(infos)` returns new scenario params when the level changes."""
from __future__ import annotations

from collections import deque


class StageCurriculum:
    """Shooter: advance stage k -> k+1 once the rolling win rate at stage k exceeds `threshold`.
    After advancing, `replay` of the probability mass is spread over earlier stages."""

    def __init__(self, n_stages: int = 5, start: int = 0, threshold: float = 0.8, window: int = 200,
                 replay: float = 0.2, min_episodes: int = 100):
        self.n, self.stage, self.thr, self.replay, self.min_eps = n_stages, start, threshold, replay, min_episodes
        self.hist: deque = deque(maxlen=window)

    def params(self) -> dict:
        p = [0.0] * self.n
        if self.stage == 0:
            p[0] = 1.0
        else:
            p[self.stage] = 1.0 - self.replay
            for i in range(self.stage):
                p[i] = self.replay / self.stage
        return {"stage_probs": p}

    def update(self, infos: list[dict]) -> dict | None:
        for d in infos:
            if int(d.get("stage", -1)) == self.stage:
                self.hist.append(float(d.get("win", 0)))
        if (self.stage < self.n - 1 and len(self.hist) >= self.min_eps
                and sum(self.hist) / len(self.hist) >= self.thr):
            self.stage += 1
            self.hist.clear()
            return self.params()
        return None

    @property
    def level(self) -> float:
        return float(self.stage)


class DifficultyCurriculum:
    """Parkour: raise difficulty by `step` when the rolling success rate exceeds `threshold`."""

    def __init__(self, start: float = 0.0, step: float = 0.1, threshold: float = 0.7, window: int = 100,
                 spread: float = 0.15, min_episodes: int = 60):
        self.d, self.step, self.thr, self.spread, self.min_eps = start, step, threshold, spread, min_episodes
        self.hist: deque = deque(maxlen=window)

    def params(self) -> dict:
        return {"difficulty_range": [max(0.0, self.d - self.spread), self.d]}

    def update(self, infos: list[dict]) -> dict | None:
        for d in infos:
            self.hist.append(float(d.get("win", 0)))
        if self.d < 1.0 and len(self.hist) >= self.min_eps and sum(self.hist) / len(self.hist) >= self.thr:
            self.d = min(1.0, self.d + self.step)
            self.hist.clear()
            return self.params()
        return None

    @property
    def level(self) -> float:
        return self.d
