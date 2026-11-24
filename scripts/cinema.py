"""High-quality "observer" videos: record a replay with a policy, re-render it in Godot cinema mode
(Forward+, HDRI, soft shadows, SSAO/SSIL, animated CC0 characters, particles), then composite.

    uv run python scripts/cinema.py --ckpt runs/T3_shooter_stage0/latest.pt --name shooter
    uv run python scripts/cinema.py --scenario hideseek --params '{"mode":"hideseek"}' --random --name hideseek_random
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from sandpit.bridge import GODOT_BIN, PROJECT_DIR, GodotProcess  # noqa: E402

FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]


def font(size: int):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def record(scenario: str, params: dict, ckpt: str, random_policy: bool, decisions: int, seed: int, replay: Path,
           focus: int, action_repeat: int, deterministic: bool) -> tuple[dict, list]:
    pol = None
    if ckpt and not random_policy:
        from sandpit.policy import PPOPolicy

        pol = PPOPolicy(ckpt)
        action_repeat = int(pol.cfg["action_repeat"])
    proc = GodotProcess(scenario, mode="watch", arenas=1, seed=seed, params=params, action_repeat=action_repeat,
                        user_args={"realtime": 0, "replay_out": str(replay), "replay_every": 2, "minimize": 1},
                        log_path=ROOT / "runs/logs" / f"cinema_record_{replay.stem}.log")
    rng = np.random.default_rng(seed)
    povs, infos = {}, []
    try:
        r = proc.reset(seed=seed)
        if pol:
            pol.reset(r.pov)
        povs[r.header["tick"]] = r.pov[focus].copy()
        for _ in range(decisions):
            if pol:
                a = pol.act(r.vec, [x["policy"] for x in proc.agents], deterministic=deterministic)
            else:
                a = np.stack([rng.integers(0, n, size=proc.n) for n in proc.nvec], 1).astype(np.int32)
                a[:, 0] = np.where(rng.random(proc.n) < 0.5, 1, a[:, 0])
            r = proc.step(a)
            if pol:
                pol.observe(r.pov, r.done)
            povs[r.header["tick"]] = r.pov[focus].copy()
            infos += r.infos
    finally:
        proc.close()
    return povs, infos


def read_replay_ticks(replay: Path) -> tuple[dict, np.ndarray]:
    with open(replay, "rb") as f:
        header = json.loads(f.readline())
        data = np.frombuffer(f.read(), dtype="<f4")
    F = 1 + header["n_nodes"] * header["node_floats"] + header["n_agents"] * header["agent_floats"]
    return header, data[: len(data) // F * F].reshape(-1, F)[:, 0].astype(int)


def render(replay: Path, out_dir: Path, cams: str, res: str, max_frames: int, focus: int, log: Path) -> None:
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    cmd = [GODOT_BIN, "--path", str(PROJECT_DIR), "--rendering-method", "forward_plus", "--windowed",
           "--resolution", "640x360", "--", "--mode=cinema", f"--replay={replay}", f"--out_dir={out_dir}",
           f"--cams={cams}", f"--res={res}", f"--max_frames={max_frames}", f"--focus={focus}", "--minimize=1"]
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=3600, check=False)
    txt = log.read_text()
    if "CINEMA_DONE" not in txt:
        raise RuntimeError(f"cinema render failed, see {log}")
    print(f"[cinema] rendered in {time.time() - t0:.0f}s: {txt.split('CINEMA_DONE')[1].splitlines()[0].strip()}")


def compose(out_dir: Path, ticks: np.ndarray, povs: dict, title: str, subtitle: str, name: str, gif_seconds: float,
            stills: int) -> None:
    import imageio.v3 as iio

    vids = ROOT / "media/videos"
    stills_dir = ROOT / "media/cinema"
    vids.mkdir(parents=True, exist_ok=True)
    stills_dir.mkdir(parents=True, exist_ok=True)
    obs = sorted(out_dir.glob("observer_*.jpg"))
    pov_ticks = np.array(sorted(povs))
    f_big, f_small = font(34), font(20)
    frames = []
    for k, p in enumerate(obs):
        base = Image.open(p).convert("RGB")
        W, H = base.size
        d = ImageDraw.Draw(base, "RGBA")
        chase_p = out_dir / p.name.replace("observer_", "chase_")
        if chase_p.exists():
            ch = Image.open(chase_p).convert("RGB").resize((W * 3 // 10, H * 3 // 10), Image.LANCZOS)
            x0, y0 = W - ch.width - 24, H - ch.height - 24
            d.rectangle([x0 - 3, y0 - 3, x0 + ch.width + 2, y0 + ch.height + 2], fill=(255, 255, 255, 230))
            base.paste(ch, (x0, y0))
            d.text((x0 + 10, y0 + 8), "CHASE CAM", font=f_small, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        t = ticks[k] if k < len(ticks) else ticks[-1]
        kk = pov_ticks[max(0, np.searchsorted(pov_ticks, t, side="right") - 1)]
        pov = Image.fromarray(povs[int(kk)]).resize((H * 2 // 9, H * 2 // 9), Image.NEAREST)
        x0, y0 = 24, H - pov.height - 24
        d.rectangle([x0 - 3, y0 - 3, x0 + pov.width + 2, y0 + pov.height + 2], fill=(255, 220, 0, 235))
        base.paste(pov, (x0, y0))
        d.text((x0 + 8, y0 - 30), "AGENT VIEW 64x64 (policy input)", font=f_small, fill=(255, 230, 0), stroke_width=2, stroke_fill=(0, 0, 0))
        d.rounded_rectangle([20, 20, 20 + max(d.textlength(title, font=f_big), d.textlength(subtitle, font=f_small)) + 36, 112],
                            radius=14, fill=(10, 12, 18, 170))
        d.text((38, 30), title, font=f_big, fill=(255, 255, 255))
        d.text((38, 76), subtitle, font=f_small, fill=(200, 210, 225))
        d.text((W - 210, 30), f"t = {t / 60:5.1f} s", font=f_small, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        frames.append(np.asarray(base))
    fps = 30
    # constant-quality H.264 keeps 1080p clips repo-sized (~5-15 MB / 20 s)
    iio.imwrite(vids / f"{name}_cinema.mp4", np.stack(frames), fps=fps, codec="libx264", quality=None,
                macro_block_size=8, ffmpeg_params=["-crf", "26", "-preset", "slow", "-pix_fmt", "yuv420p"])
    n_gif = min(len(frames), int(gif_seconds * fps))
    small = [np.asarray(Image.fromarray(f).resize((640, 360), Image.LANCZOS)) for f in frames[:n_gif:3]]
    iio.imwrite(vids / f"{name}_cinema.gif", np.stack(small), duration=1000 * 3 / fps, loop=0)
    for j, idx in enumerate(np.linspace(len(frames) * 0.15, len(frames) - 1, stills).astype(int)):
        Image.fromarray(frames[idx]).resize((1600, 900), Image.LANCZOS).save(stills_dir / f"{name}_{j}.jpg", quality=88)
    print(f"[cinema] wrote media/videos/{name}_cinema.mp4 ({len(frames)} frames), .gif and {stills} stills in media/cinema/")


def main(name: str, ckpt: str = "", scenario: str = "", params: str = "", random: bool = False, decisions: int = 300,
         seed: int = 7, focus: int = 0, cams: str = "observer,chase", res: str = "1920x1080", max_frames: int = 100000,
         title: str = "", subtitle: str = "", gif_seconds: float = 6.0, stills: int = 3, deterministic: bool = False,
         action_repeat: int = 4):
    cfg = {}
    if ckpt:
        import torch

        cfg = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]
    scen = scenario or cfg.get("scenario", "shooter")
    prm = json.loads(params) if params else json.loads(cfg.get("params", "{}"))
    work = ROOT / "runs/cinema" / name
    work.mkdir(parents=True, exist_ok=True)
    replay = work / "replay.bin"
    povs, infos = record(scen, prm, ckpt, random, decisions, seed, replay, focus, action_repeat, deterministic)
    print("[cinema] episodes:", json.dumps([{k: d[k] for k in d if k in ("agent", "ep_return", "win", "kills", "platforms", "result", "tags", "hidden_frac")} for d in infos])[:600])
    _, ticks = read_replay_ticks(replay)
    render(replay, work / "frames", cams, res, max_frames, focus, work / "render.log")
    who = "random policy" if (random or not ckpt) else f"PPO checkpoint {Path(ckpt).parent.name}"
    compose(work / "frames", ticks, povs, title or f"{scen.upper()}", subtitle or f"{who} | replay re-rendered in Godot 4.7 Forward+",
            name, gif_seconds, stills)


if __name__ == "__main__":
    tyro.cli(main)
