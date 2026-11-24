"""Throughput benchmark: agent-steps/s per scenario vs number of Godot processes (random actions)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tyro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.vec_env import GodotVecEnv  # noqa: E402
from sandpit.viz import plot_curves  # noqa: E402

SCEN = {
    "shooter": ("shooter", {"stage": 3}, 8),
    "sumo": ("sumo", {}, 4),
    "tag": ("tag", {"mode": "tag"}, 4),
    "hideseek": ("hideseek", {"mode": "hideseek"}, 2),
    "parkour": ("parkour", {"difficulty": 0.5}, 8),
}


def run(scen: str, params: dict, procs: int, arenas: int, steps: int) -> float:
    env = GodotVecEnv(scen, n_procs=procs, arenas_per_proc=arenas, params=params, log_dir=ROOT / "runs/logs")
    rng = np.random.default_rng(0)
    try:
        env.reset(seed=0)
        for _ in range(20):
            env.step(env.random_actions(rng))
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(env.random_actions(rng))
        return steps * env.num_agents / (time.perf_counter() - t0)
    finally:
        env.close()


def main(names: tuple[str, ...] = tuple(SCEN), procs: tuple[int, ...] = (1, 2, 4, 6), steps: int = 300):
    res = {}
    for n in names:
        scen, params, arenas = SCEN[n]
        res[n] = {}
        for p in procs:
            sps = run(scen, params, p, arenas, steps)
            res[n][p] = round(sps)
            print(f"{n:9s} procs={p} arenas/proc={arenas} -> {sps:7.0f} agent-steps/s", flush=True)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/bench.json").write_text(json.dumps(res, indent=1))
    plot_curves({k: (list(v.keys()), list(v.values())) for k, v in res.items()}, ROOT / "media/screenshots/bench_throughput.png",
                title="Environment throughput (64x64 RGB observations, random actions)", xlabel="Godot processes",
                ylabel="agent-steps / s")


if __name__ == "__main__":
    tyro.cli(main)
