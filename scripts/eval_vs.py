"""Evaluate a trained checkpoint against random-action opponents (absolute check for self-play runs).

    sumo: team 0 uses the checkpoint, team 1 acts uniformly at random -> team-0 win rate
    tag : --learner-team selects which side uses the checkpoint (0 hider, 1 seeker)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tyro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.policy import PPOPolicy  # noqa: E402
from sandpit.vec_env import GodotVecEnv  # noqa: E402


def main(ckpt: str, decisions: int = 400, procs: int = 2, arenas: int = 8, learner_team: int = 0, seed: int = 123,
         both_trained: bool = False, out: str = ""):
    pol = PPOPolicy(ckpt)
    cfg = pol.cfg
    env = GodotVecEnv(cfg["scenario"], n_procs=procs, arenas_per_proc=arenas, seed=seed, params=json.loads(cfg["params"]),
                      action_repeat=int(cfg["action_repeat"]), log_dir=ROOT / "runs/logs")
    rng = np.random.default_rng(seed)
    teams = np.array([m.team for m in env.meta])
    pols = [m.policy for m in env.meta]
    res = []
    try:
        obs = env.reset(seed=seed)
        pol.reset(obs.pov)
        for _ in range(decisions):
            a = pol.act(obs.vec, pols)
            if not both_trained:
                rnd = env.random_actions(rng)
                a[teams != learner_team] = rnd[teams != learner_team]
            obs = env.step(a)
            pol.observe(obs.pov, obs.done)
            for d in obs.infos:
                if teams[int(d["agent"])] == learner_team:
                    res.append(d)
    finally:
        env.close()
    win = float(np.mean([d.get("win", 0) for d in res])) if res else float("nan")
    summary = {"ckpt": ckpt, "learner_team": learner_team, "episodes": len(res), "win_rate_vs_random": win,
               "mean_return": float(np.mean([d["ep_return"] for d in res])) if res else float("nan")}
    if res and "result" in res[0]:
        r = np.array([d["result"] for d in res])
        summary.update({"wins": int((r > 0).sum()), "losses": int((r < 0).sum()), "draws": int((r == 0).sum())})
    print(json.dumps(summary))
    if out:
        Path(out).write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    tyro.cli(main)
