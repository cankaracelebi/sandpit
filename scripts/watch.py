"""Watch / record a trained PPO policy in Godot.

Fast offline recording (default): Godot renders overview + chase cameras every `record_every`
ticks; Python composites them with the agent POV into MP4 + GIF under media/videos/.
Live viewing: --realtime (opens a normal window at 60 Hz).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.bridge import GodotProcess  # noqa: E402
from sandpit.policy import PPOPolicy  # noqa: E402


def compose(overview: Image.Image, chase: Image.Image, pov: np.ndarray, text: str) -> np.ndarray:
    h = 360
    pov_im = Image.fromarray(pov).resize((h, h), Image.NEAREST)
    W = overview.width + chase.width + h + 16
    W, Ht = (W + 15) // 16 * 16, (h + 28 + 15) // 16 * 16  # ffmpeg macro blocks
    canvas = Image.new("RGB", (W, Ht), (18, 18, 22))
    canvas.paste(overview, (4, 24))
    canvas.paste(chase, (overview.width + 8, 24))
    canvas.paste(pov_im, (overview.width + chase.width + 12, 24))
    d = ImageDraw.Draw(canvas)
    d.text((6, 5), text, fill=(235, 235, 235))
    d.text((overview.width + chase.width + 14, 26), "agent POV 64x64 (policy input)", fill=(255, 255, 0))
    return np.asarray(canvas)


def main(ckpt: str, decisions: int = 300, scenario: str = "", params: str = "", name: str = "", seed: int = 7,
         realtime: bool = False, record_every: int = 2, deterministic: bool = False, gif_seconds: float = 8.0,
         gif_scale: float = 0.5):
    pol = PPOPolicy(ckpt)
    scen = scenario or pol.cfg["scenario"]
    prm = json.loads(params) if params else json.loads(pol.cfg["params"])
    if pol.cfg.get("curriculum") == "difficulty" and "difficulty" not in prm:
        prm.setdefault("difficulty", 0.3)
    name = name or f"{scen}_{Path(ckpt).parent.name}"
    frames_dir = ROOT / "runs/watch_frames" / name
    shutil.rmtree(frames_dir, ignore_errors=True)
    proc = GodotProcess(scen, mode="watch", arenas=1, seed=seed, params=prm, window=(960, 540),
                        action_repeat=int(pol.cfg["action_repeat"]),
                        user_args={"realtime": int(realtime), "record_dir": "" if realtime else str(frames_dir),
                                   "record_every": record_every, "minimize": 0 if realtime else 1},
                        log_path=ROOT / "runs/logs" / f"watch_{name}.log")
    povs: dict[int, np.ndarray] = {}
    infos = []
    try:
        r = proc.reset(seed=seed)
        pol.reset(r.pov)
        povs[r.header["tick"]] = r.pov[0].copy()
        for _ in range(decisions):
            a = pol.act(r.vec, [x["policy"] for x in proc.agents], deterministic=deterministic)
            r = proc.step(a)
            pol.observe(r.pov, r.done)
            povs[r.header["tick"]] = r.pov[0].copy()
            infos += r.infos
    finally:
        proc.close()
    print("episodes:", json.dumps([{k: v for k, v in d.items() if k in ("agent", "ep_return", "win", "kills", "platforms", "result", "tags", "hidden_frac")} for d in infos])[:800])
    if realtime:
        return
    ticks = sorted({int(p.name.split("_")[0]) for p in frames_dir.glob("*_overview.png")})
    pov_ticks = np.array(sorted(povs))
    frames = []
    for t in ticks:
        k = pov_ticks[np.searchsorted(pov_ticks, t, side="right") - 1]
        ov = Image.open(frames_dir / f"{t:06d}_overview.png").convert("RGB")
        ch = Image.open(frames_dir / f"{t:06d}_chase.png").convert("RGB")
        frames.append(compose(ov, ch, povs[int(k)], f"{scen} | {Path(ckpt).parent.name} | t={t / 60:.1f}s"))
    import imageio.v3 as iio

    out_dir = ROOT / "media/videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 60 // record_every
    iio.imwrite(out_dir / f"{name}.mp4", np.stack(frames), fps=fps, codec="libx264", quality=7)
    n_gif = int(gif_seconds * fps)
    small = [np.asarray(Image.fromarray(f).resize((int(f.shape[1] * gif_scale), int(f.shape[0] * gif_scale)), Image.BILINEAR))
             for f in frames[:n_gif:2]]
    iio.imwrite(out_dir / f"{name}.gif", np.stack(small), duration=1000 * 2 / fps, loop=0)
    Image.fromarray(frames[len(frames) // 2]).save(out_dir / f"{name}_still.png")
    print(f"wrote {out_dir / (name + '.mp4')} ({len(frames)} frames) and .gif")


if __name__ == "__main__":
    tyro.cli(main)
