"""Dreamer drivers: dataset recording, offline world-model training, and online Dreamer RL."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..ppo import pick_device
from ..vec_env import GodotVecEnv
from ..viz import plot_curves
from .agent import Dreamer, DreamerConfig
from .replay import StreamReplay, StreamWriter

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class DreamerRunConfig:
    mode: str = "online"            # record | wm | online
    scenario: str = "shooter"
    params: str = '{"stage": 0}'
    procs: int = 4
    arenas: int = 8
    seed: int = 1
    device: str = "auto"
    run_name: str = ""
    # data
    data: str = ""                  # npz dataset (record output / wm input)
    ppo_ckpt: str = ""              # record with this PPO policy (else random)
    epsilon: float = 0.3            # random-action prob while recording with a policy
    record_steps: int = 600         # decisions to record
    capacity: int = 3000            # rows per agent stream
    # training
    batch: int = 16
    seq_len: int = 48
    wm_steps: int = 3000            # wm mode: gradient steps
    env_steps: int = 100_000        # online mode: agent-steps budget
    prefill: int = 100              # online: random decisions before training
    train_every: int = 2            # online: decisions between training rounds
    train_iters: int = 1            # online: gradient steps per round
    log_every: int = 100
    strip_every: int = 1000
    model: DreamerConfig = field(default_factory=DreamerConfig)


def _run_dir(cfg: DreamerRunConfig) -> Path:
    name = cfg.run_name or f"dreamer_{cfg.mode}_{cfg.scenario}_{time.strftime('%m%d_%H%M%S')}"
    d = ROOT / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    return d


def save_strip(gt: np.ndarray, pred: np.ndarray, path: Path, context: int, n: int = 3, every: int = 2) -> Path:
    """Rows: ground truth / model (context = posterior reconstruction, rest = open-loop imagination)."""
    from PIL import Image, ImageDraw

    L = gt.shape[1]
    ts = list(range(0, L, every))
    s, pad = 64, 2
    W = len(ts) * (s + pad) + 70
    Hh = n * (2 * s + 3 * pad + 8) + 20
    canvas = Image.new("RGB", (W, Hh), (20, 20, 24))
    d = ImageDraw.Draw(canvas)
    d.text((4, 4), f"top: ground truth | bottom: world model  (t < {context}: observed, t >= {context}: imagined open-loop)", fill=(230, 230, 230))
    y = 20
    for k in range(min(n, gt.shape[0])):
        d.text((4, y + 24), "GT", fill=(200, 200, 200))
        d.text((4, y + s + 28), "WM", fill=(200, 200, 200))
        for j, t in enumerate(ts):
            x = 70 + j * (s + pad)
            canvas.paste(Image.fromarray(gt[k, t]), (x, y))
            canvas.paste(Image.fromarray(pred[k, t]), (x, y + s + pad))
            if t >= context:
                d.rectangle([x, y + s + pad, x + s - 1, y + 2 * s + pad - 1], outline=(255, 170, 0))
        y += 2 * s + 3 * pad + 8
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


# ---------------------------------------------------------------------- record
def record(cfg: DreamerRunConfig) -> Path:
    run = _run_dir(cfg)
    env = GodotVecEnv(cfg.scenario, n_procs=cfg.procs, arenas_per_proc=cfg.arenas, seed=cfg.seed,
                      params=json.loads(cfg.params), log_dir=run / "logs")
    rng = np.random.default_rng(cfg.seed)
    policy = None
    if cfg.ppo_ckpt:
        from ..policy import PPOPolicy

        policy = PPOPolicy(cfg.ppo_ckpt, cfg.device)
    ids = np.arange(env.num_agents)
    rep = StreamReplay(len(ids), cfg.capacity, env.pov_shape, env.vec_dim, len(env.nvec), cfg.seed)
    w = StreamWriter(rep, ids)
    try:
        obs = env.reset(seed=cfg.seed)
        w.start(obs.pov, obs.vec)
        if policy:
            policy.reset(obs.pov)
        vec = obs.vec
        for _ in range(cfg.record_steps):
            a = env.random_actions(rng)
            if policy:
                pa = policy.act(vec, [m.policy for m in env.meta])
                use = rng.random(env.num_agents) > cfg.epsilon
                a[use] = pa[use]
            obs = env.step(a)
            w.step(a, obs.pov, obs.vec, obs.reward, obs.done, obs.trunc)
            if policy:
                policy.observe(obs.pov, obs.done)
            vec = obs.vec
    finally:
        env.close()
    out = Path(cfg.data) if cfg.data else run / "dataset.npz"
    rep.save(out)
    print(f"[dreamer] recorded {rep.size} rows from {len(ids)} streams -> {out}", flush=True)
    return out


# ---------------------------------------------------------------------- offline world model
def train_wm(cfg: DreamerRunConfig) -> Path:
    run = _run_dir(cfg)
    dev = pick_device(cfg.device)
    rep = StreamReplay.load(cfg.data, cfg.seed)
    nvec = (3, 3, 5, 3, 2, 2)
    agent = Dreamer(nvec, rep.vec.shape[-1], cfg.model).to(dev)
    hist = defaultdict(list)
    t0 = time.perf_counter()
    for it in range(1, cfg.wm_steps + 1):
        m, _ = agent.train_wm(rep.sample(cfg.batch, cfg.seq_len), dev)
        for k, v in m.items():
            hist[k].append(v)
        if it % cfg.log_every == 0:
            print(f"[wm] it {it} " + " ".join(f"{k.split('/')[1]}={np.mean(v[-cfg.log_every:]):.3f}" for k, v in hist.items())
                  + f" ({(time.perf_counter() - t0) / it:.3f}s/it)", flush=True)
        if it % cfg.strip_every == 0 or it == cfg.wm_steps:
            gt, pred = agent.openloop(rep.sample(4, 32), dev, context=8)
            save_strip(gt, pred, run / f"openloop_{it:05d}.png", context=8)
    its = np.arange(1, cfg.wm_steps + 1)
    plot_curves({"image recon": (its, hist["wm/rec_img"])}, run / "wm_loss.png", title="World model: image reconstruction loss",
                xlabel="gradient steps", ylabel="sum sq. error / frame", smooth=50)
    plot_curves({k.split("/")[1]: (its, v) for k, v in hist.items() if k in ("wm/rew", "wm/cont", "wm/kl")},
                run / "wm_aux_losses.png", title="World model: reward / continue / KL", xlabel="gradient steps", smooth=50)
    torch.save({"model": agent.state_dict(), "cfg": asdict(cfg)}, run / "wm.pt")
    (run / "wm_history.json").write_text(json.dumps({k: v[:: max(1, len(v) // 500)] for k, v in hist.items()}))
    return run


# ---------------------------------------------------------------------- online Dreamer
def train_online(cfg: DreamerRunConfig) -> Path:
    run = _run_dir(cfg)
    dev = pick_device(cfg.device)
    env = GodotVecEnv(cfg.scenario, n_procs=cfg.procs, arenas_per_proc=cfg.arenas, seed=cfg.seed,
                      params=json.loads(cfg.params), log_dir=run / "logs")
    rng = np.random.default_rng(cfg.seed)
    N = env.num_agents
    ids = np.arange(N)
    rep = StreamReplay(N, cfg.capacity, env.pov_shape, env.vec_dim, len(env.nvec), cfg.seed)
    writer = StreamWriter(rep, ids)
    agent = Dreamer(env.nvec, env.vec_dim, cfg.model).to(dev)
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(run / "tb"))
    ep_log = open(run / "episodes.jsonl", "a")
    step, decisions, grad_steps = 0, 0, 0
    hist = defaultdict(list)
    t0 = time.perf_counter()
    try:
        obs = env.reset(seed=cfg.seed)
        writer.start(obs.pov, obs.vec)
        state, prev_a = None, torch.zeros((N, 6), dtype=torch.long, device=dev)
        first = torch.ones(N, dtype=torch.bool, device=dev)
        while step < cfg.env_steps:
            if decisions < cfg.prefill:
                a = env.random_actions(rng)
            else:
                at, state = agent.policy_step(state, prev_a, torch.from_numpy(obs.pov).to(dev),
                                              torch.from_numpy(obs.vec).to(dev), first)
                a = at.cpu().numpy().astype(np.int32)
            obs = env.step(a)
            writer.step(a, obs.pov, obs.vec, obs.reward, obs.done, obs.trunc)
            prev_a = torch.from_numpy(a.astype(np.int64)).to(dev)
            first = torch.from_numpy(obs.done).to(dev)
            prev_a[first] = 0
            decisions += 1
            step += N
            for d in obs.infos:
                ep_log.write(json.dumps({"step": step, "policy": env.meta[int(d["agent"])].policy, **d}) + "\n")
                for k in ("ep_return", "win", "kills", "platforms"):
                    if k in d:
                        hist[f"episode/{k}"].append((step, float(d[k])))
                        tb.add_scalar(f"episode/{k}", float(d[k]), step)
            if decisions >= cfg.prefill and decisions % cfg.train_every == 0:
                for _ in range(cfg.train_iters):
                    m, starts = agent.train_wm(rep.sample(cfg.batch, cfg.seq_len), dev)
                    m.update(agent.train_ac(starts))
                    grad_steps += 1
                    if grad_steps % 20 == 0:
                        for k, v in m.items():
                            tb.add_scalar(k, v, step)
                            hist[k].append((step, v))
                if grad_steps % cfg.log_every == 0:
                    rec = [v for s, v in hist.get("episode/ep_return", [])[-50:]]
                    kills = [v for s, v in hist.get("episode/kills", [])[-50:]]
                    print(f"[dreamer] step={step} grad={grad_steps} sps={int(step / (time.perf_counter() - t0))} "
                          f"ret50={np.mean(rec) if rec else float('nan'):.3f} kills50={np.mean(kills) if kills else float('nan'):.3f} "
                          f"rec_img={m['wm/rec_img']:.1f} kl={m['wm/kl']:.2f} imag_ret={m['ac/imag_return']:.3f}", flush=True)
                if grad_steps % cfg.strip_every == 0:
                    gt, pred = agent.openloop(rep.sample(4, 32), dev, context=8)
                    save_strip(gt, pred, run / f"openloop_{grad_steps:05d}.png", context=8)
        gt, pred = agent.openloop(rep.sample(4, 32), dev, context=8)
        save_strip(gt, pred, run / "openloop_final.png", context=8)
        torch.save({"model": agent.state_dict(), "cfg": asdict(cfg)}, run / "dreamer.pt")
        (run / "history.json").write_text(json.dumps({k: v[:: max(1, len(v) // 1000)] for k, v in hist.items()}))
    finally:
        env.close()
        tb.close()
        ep_log.close()
    return run
