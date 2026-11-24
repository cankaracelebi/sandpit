"""Capture spectator / chase / top-down / agent-POV screenshots for scenarios.

Runs Godot in `screenshot` mode (random actions for `ticks` physics ticks, then capture) and
composes a labelled overview image per scenario into media/screenshots/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import tyro
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.bridge import GODOT_BIN, PROJECT_DIR  # noqa: E402
from sandpit.viz import compose_row, pov_grid_image  # noqa: E402

PRESETS = {
    "shooter_s0": ("shooter", {"stage": 0}),
    "shooter_s2": ("shooter", {"stage": 2}),
    "shooter_s4": ("shooter", {"stage": 4}),
    "sumo": ("sumo", {}),
    "tag": ("tag", {"mode": "tag"}),
    "hideseek": ("hideseek", {"mode": "hideseek"}),
    "parkour_easy": ("parkour", {"difficulty": 0.2}),
    "parkour_hard": ("parkour", {"difficulty": 0.9}),
}


def capture(name: str, scenario: str, params: dict, ticks: int = 150, seed: int = 3, shadows: bool = True,
            out_dir: Path = ROOT / "media/screenshots", pov: int = 64) -> Path:
    raw = ROOT / "runs/shots_raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    cmd = [GODOT_BIN, "--path", str(PROJECT_DIR), "--windowed", "--resolution", "320x240", "--",
           "--mode=screenshot", f"--scenario={scenario}", f"--seed={seed}", "--arenas=1", f"--pov={pov}",
           f"--shadows={int(shadows)}", f"--shot_dir={raw}", f"--shot_ticks={ticks}", f"--shot_name={name}",
           "--minimize=1", "--params=" + urllib.parse.quote(json.dumps(params))]
    log = ROOT / "runs/logs" / f"shot_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=180, check=False)
    if "SCREENSHOTS_DONE" not in log.read_text():
        raise RuntimeError(f"screenshot run failed for {name}; see {log}")
    povs = sorted((p for p in raw.glob(f"{name}_pov*.png") if p.stem.split("pov")[-1].isdigit()), key=lambda p: int(p.stem.split("pov")[-1]))
    pov_arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in povs])
    pov_path = raw / f"{name}_povgrid.png"
    pov_grid_image(pov_arr, scale=4, cols=min(len(povs), 2), labels=[f"agent {i} POV {pov}x{pov}" for i in range(len(povs))]).save(pov_path)
    out = compose_row([raw / f"{name}_spectator.png", raw / f"{name}_chase.png", raw / f"{name}_topdown.png", pov_path],
                      out_dir / f"{name}.png", height=300,
                      title=f"{name.split("_")[0].capitalize()}: spectator | chase camera | top-down | agent observations (64x64)")
    return out


def main(names: tuple[str, ...] = (), ticks: int = 150, seed: int = 3):
    names = names or tuple(PRESETS)
    for n in names:
        scen, params = PRESETS[n]
        print("captured", capture(n, scen, params, ticks=ticks, seed=seed))


if __name__ == "__main__":
    tyro.cli(main)
