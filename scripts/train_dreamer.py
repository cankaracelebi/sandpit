"""Dreamer entry point.
    record : python scripts/train_dreamer.py --mode record --ppo-ckpt runs/T3_shooter_stage0/latest.pt --data runs/shooter_data.npz
    wm     : python scripts/train_dreamer.py --mode wm --data runs/shooter_data.npz --wm-steps 3000
    online : python scripts/train_dreamer.py --mode online --env-steps 100000
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import tyro  # noqa: E402

from sandpit.dreamer.train import DreamerRunConfig, record, train_online, train_wm  # noqa: E402

if __name__ == "__main__":
    cfg = tyro.cli(DreamerRunConfig)
    {"record": record, "wm": train_wm, "online": train_online}[cfg.mode](cfg)
