"""Phase-1 bridge spike: frame freshness, non-blank pixels, throughput, POV grid screenshot."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))  # robust to hidden .pth files

import numpy as np
import tyro

from sandpit.vec_env import GodotVecEnv
from sandpit.viz import save_pov_grid

ROOT = Path(__file__).resolve().parents[1]


def main(procs: int = 1, arenas: int = 2, agents_per_arena: int = 4, steps: int = 300, action_repeat: int = 4,
         shadows: bool = False, pov: int = 64, out: str = "media/screenshots/spike_pov_grid.png",
         profile: bool = False):
    env = GodotVecEnv("spike", n_procs=procs, arenas_per_proc=arenas, action_repeat=action_repeat, shadows=shadows,
                      pov=pov, params={"agents_per_arena": agents_per_arena}, log_dir=ROOT / "runs/logs",
                      user_args={"profile": 1} if profile else None)
    rng = np.random.default_rng(0)
    try:
        obs = env.reset(seed=1)
        mism = 0
        checks = 0
        stds = []
        dones = 0
        t0 = time.perf_counter()
        for step in range(steps):
            obs = env.step(env.random_actions(rng))
            h, w = obs.pov.shape[1:3]
            px = obs.pov[:, h // 2, w // 2].astype(np.float32)
            # only score agents whose tick quad is actually visible (G~40, B~200); a wall closer
            # than the quad legitimately occludes it
            vis = (np.abs(px[:, 1] - 40) < 12) & (np.abs(px[:, 2] - 200) < 12)
            level = np.clip(np.round((px[:, 0] - 16.0) / 32.0), 0, 7)
            expected = obs.vec[:, -1]
            mism += int(((level != expected) & vis).sum())
            checks += int(vis.sum())
            stds.append(obs.pov.reshape(len(obs.pov), -1).std(axis=1).mean())
            dones += int(obs.done.sum())
            if step == steps // 2:
                save_pov_grid(obs.pov, ROOT / out, title=f"spike POVs ({env.num_agents} agents)")
        dt = time.perf_counter() - t0
        print(f"agents={env.num_agents} steps={steps} wall={dt:.2f}s "
              f"decisions/s={steps / dt:.1f} agent-steps/s={steps * env.num_agents / dt:.0f}")
        print(f"freshness mismatches: {mism}/{checks}  mean pixel std: {np.mean(stds):.1f}  episode ends: {dones}")
        return mism, checks
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
