"""Plot learning curves from a run's episodes.jsonl (binned by env steps), with random baselines."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tyro

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.viz import plot_curves  # noqa: E402


def load_episodes(run: Path) -> list[dict]:
    return [json.loads(l) for l in open(run / "episodes.jsonl") if l.strip()]


def binned(eps: list[dict], key: str, policy: str | None, bins: int) -> tuple[np.ndarray, np.ndarray]:
    pts = [(e["step"], float(e[key])) for e in eps if key in e and isinstance(e[key], (int, float))
           and (policy is None or e.get("policy") == policy)]
    if not pts:
        return np.array([]), np.array([])
    s, v = np.array(pts).T
    edges = np.linspace(0, s.max() + 1, bins + 1)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (s >= a) & (s < b)
        if m.sum() >= 3:
            xs.append((a + b) / 2)
            ys.append(v[m].mean())
    return np.array(xs), np.array(ys)


def main(run: str, keys: tuple[str, ...] = ("ep_return",), policies: tuple[str, ...] = (), bins: int = 30,
         out_prefix: str = "", title: str = ""):
    run_p = Path(run) if Path(run).is_absolute() else ROOT / run
    eps = load_episodes(run_p)
    base = {}
    if (run_p / "baseline_random.json").exists():
        base = json.loads((run_p / "baseline_random.json").read_text())
    pols = list(policies) or sorted({e.get("policy") for e in eps})
    outs = []
    for k in keys:
        series = {}
        for p in pols:
            x, y = binned(eps, k, p, bins)
            if len(x):
                series[f"{p} (PPO)" if len(pols) > 1 else "PPO"] = (x, y)
        b = base.get(f"{pols[0]}/{k}") if len(pols) == 1 else None
        out = ROOT / "media/screenshots" / f"{out_prefix or run_p.name}_{k}.png"
        plot_curves(series, out, title=title or f"{run_p.name}: {k}", ylabel=k, baseline=b,
                    baseline_label="random policy")
        outs.append(out)
        print(out)
    return outs


if __name__ == "__main__":
    tyro.cli(main)
