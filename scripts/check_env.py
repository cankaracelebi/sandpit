"""Smoke-test scenarios through the real vectorised env with random actions.

Checks: obs shapes and dtypes, finite rewards/vec, episodes end and auto-reset, POV images not
blank, and well-formed episode infos. Saves a POV grid per scenario and prints a summary.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tyro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.vec_env import GodotVecEnv  # noqa: E402
from sandpit.viz import save_pov_grid  # noqa: E402

CONFIGS = {
    "shooter": ("shooter", {"stage_probs": [0.2, 0.2, 0.2, 0.2, 0.2]}),
    "sumo": ("sumo", {}),
    "tag": ("tag", {"mode": "tag", "seconds": 8}),
    "hideseek": ("hideseek", {"mode": "hideseek", "seconds": 8}),
    "parkour": ("parkour", {"difficulty_range": [0.0, 1.0], "seconds": 10}),
}


def check(name: str, steps: int, procs: int, arenas: int, seed: int = 0) -> dict:
    scen, params = CONFIGS[name]
    env = GodotVecEnv(scen, n_procs=procs, arenas_per_proc=arenas, params=params, seed=seed, log_dir=ROOT / "runs/logs")
    rng = np.random.default_rng(seed)
    out: dict = {"scenario": name, "agents": env.num_agents, "policies": env.policies, "vec_dim": env.vec_dim}
    try:
        obs = env.reset(seed=seed)
        assert obs.pov.shape == (env.num_agents, *env.pov_shape) and obs.pov.dtype == np.uint8
        assert obs.vec.shape == (env.num_agents, env.vec_dim) and obs.vec.dtype == np.float32
        ep_infos, ret_sum, stds = [], 0.0, []
        per_policy_ret = defaultdict(list)
        t0 = time.perf_counter()
        for step in range(steps):
            obs = env.step(env.random_actions(rng))
            assert np.isfinite(obs.reward).all(), "non-finite reward"
            assert np.isfinite(obs.vec).all(), "non-finite vec obs"
            ret_sum += float(obs.reward.sum())
            # blind agents (hide&seek prep) legitimately see black; exclude them from blank check
            s = obs.pov.reshape(env.num_agents, -1).std(axis=1)
            stds.append(np.median(s))
            for d in obs.infos:
                assert "ep_return" in d and "ep_len" in d, d
                ep_infos.append(d)
                per_policy_ret[d.get("policy", "?")].append(d["ep_return"])
            if step == steps // 2:
                save_pov_grid(obs.pov[: min(16, env.num_agents)], ROOT / f"media/screenshots/check_{name}_povs.png",
                              title=f"{name}: agent POVs mid-run (random policy)")
        dt = time.perf_counter() - t0
        out.update({
            "agent_steps_per_s": round(steps * env.num_agents / dt),
            "episodes": len(ep_infos),
            "median_pov_std": round(float(np.median(stds)), 1),
            "mean_ep_return": {k: round(float(np.mean(v)), 3) for k, v in per_policy_ret.items()},
        })
        keys = set().union(*[d.keys() for d in ep_infos]) if ep_infos else set()
        for k in ("win", "kills", "stage", "platforms", "hidden_frac", "tags", "result", "difficulty"):
            if k in keys:
                vals = [float(d[k]) for d in ep_infos if k in d and not isinstance(d[k], (list, dict))]
                if vals:
                    out[f"mean_{k}"] = round(float(np.mean(vals)), 3)
        assert len(ep_infos) > 0, "no episode finished"
        assert np.median(stds) > 5.0, "POV images look blank"
        out["ok"] = True
    finally:
        env.close()
    return out


def main(names: tuple[str, ...] = tuple(CONFIGS), steps: int = 600, procs: int = 2, arenas: int = 2):
    results = []
    for n in names:
        r = check(n, steps, procs, arenas)
        print(json.dumps(r))
        results.append(r)
    path = ROOT / "runs/check_env.json"
    path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
