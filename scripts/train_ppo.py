"""Train PPO / self-play PPO. Usage:
    python scripts/train_ppo.py --config configs/shooter_baseline.yaml [--total-steps 100000 ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import tyro  # noqa: E402
import yaml  # noqa: E402

from sandpit.ppo import PPOConfig, PPOTrainer  # noqa: E402


def main() -> None:
    argv = sys.argv[1:]
    default = PPOConfig()
    if "--config" in argv:
        i = argv.index("--config")
        data = yaml.safe_load(open(argv[i + 1]))
        if isinstance(data.get("params"), dict):
            import json

            data["params"] = json.dumps(data["params"])
        default = PPOConfig(**data)
        argv = argv[:i] + argv[i + 2 :]
    cfg = tyro.cli(PPOConfig, default=default, args=argv)
    PPOTrainer(cfg).train()


if __name__ == "__main__":
    main()
