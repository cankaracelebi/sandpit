#!/usr/bin/env bash
# UNTESTED. Usage: cloud/run_full.sh configs/shooter_large.yaml [extra train_ppo.py args]
# Starts a virtual X server (Godot needs a display to create its window; rendering itself is
# offscreen into SubViewports), sanity-checks rendering with the bridge tests, then trains.
set -euo pipefail
CFG=${1:-configs/shooter_large.yaml}; shift || true
Xvfb :99 -screen 0 1280x1024x24 >/tmp/xvfb.log 2>&1 &
sleep 2
vulkaninfo --summary | head -20 || echo "WARNING: vulkaninfo failed (no GPU Vulkan ICD?)"
uv run pytest -q tests/test_protocol.py   # must pass: pixels non-blank + frame freshness
uv run python scripts/train_ppo.py --config "$CFG" --device cuda "$@"
