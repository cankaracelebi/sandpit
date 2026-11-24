"""Multi-policy PPO with self-play for the Godot arena (CleanRL-style, one file).

* Every policy name reported by the env (e.g. "shooter", "sumo", "hider"/"seeker") gets its own
  ActorCritic. Agents of that policy are learners unless `selfplay` names the policy, in which
  case one agent per arena is controlled by a frozen snapshot drawn from an opponent pool
  (80% latest / 20% older) and Elo is tracked.
* Observations: 4 stacked 64x64 RGB frames + the latest vector obs. Same-step auto-reset: a
  `done` at step t means the transition t ended the episode and the next obs starts a new one.
* Logs: TensorBoard + episodes.jsonl (every learner episode end) + periodic POV grids.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .curriculum import DifficultyCurriculum, StageCurriculum
from .models import ActorCritic
from .selfplay import Elo, OpponentPool, clone_module
from .vec_env import GodotVecEnv
from .viz import save_pov_grid

ROOT = Path(__file__).resolve().parents[2]
LOG_KEYS = ("ep_return", "ep_len", "win", "kills", "hits", "shots", "damage", "died", "stage", "platforms",
            "fell", "difficulty", "button", "hidden_frac", "tags", "grabs", "result", "shove_hits")


@dataclass
class PPOConfig:
    scenario: str = "shooter"
    params: str = "{}"              # JSON scenario params, e.g. '{"stage": 0}'
    procs: int = 4                  # Godot processes
    arenas: int = 8                 # arenas per process
    total_steps: int = 300_000      # learner agent-steps
    rollout: int = 64               # decisions per update
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    frame_stack: int = 4
    pov: int = 64
    action_repeat: int = 4
    norm_reward: bool = True
    selfplay: str = ""              # policy trained against its own snapshots (e.g. "sumo")
    snapshot_every: int = 10        # updates between pool snapshots / opponent resampling
    curriculum: str = ""            # "" | "stage" (shooter) | "difficulty" (parkour)
    baseline_steps: int = 0         # decisions of random play before training (baseline metrics)
    device: str = "auto"            # auto | cpu | mps | cuda
    seed: int = 1
    run_name: str = ""
    save_every: int = 10
    pov_png_every: int = 10


class RunningMeanStd:
    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = float(np.mean(x)), float(np.var(x)), x.size
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


class PPOTrainer:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.device = pick_device(cfg.device)
        name = cfg.run_name or f"ppo_{cfg.scenario}_{time.strftime('%m%d_%H%M%S')}"
        self.run_dir = ROOT / "runs" / name
        (self.run_dir / "povs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        params = json.loads(cfg.params)
        self.curr = None
        if cfg.curriculum == "stage":
            self.curr = StageCurriculum(start=int(params.pop("stage", 0)))
        elif cfg.curriculum == "difficulty":
            self.curr = DifficultyCurriculum(start=float(params.pop("difficulty", 0.0)))
        if self.curr is not None:
            params.update(self.curr.params())
        self.env = GodotVecEnv(cfg.scenario, n_procs=cfg.procs, arenas_per_proc=cfg.arenas, seed=cfg.seed,
                               action_repeat=cfg.action_repeat, pov=cfg.pov, params=params,
                               log_dir=self.run_dir / "logs")
        env = self.env
        self.N = env.num_agents
        H, W, _ = env.pov_shape
        self.K = cfg.frame_stack
        self.nets = {p: ActorCritic(env.nvec, env.vec_dim, cfg.frame_stack, H).to(self.device) for p in env.policies}
        self.opts = {p: torch.optim.Adam(n.parameters(), lr=cfg.lr, eps=1e-5) for p, n in self.nets.items()}
        pol = np.array([m.policy for m in env.meta])
        learner = np.ones(self.N, dtype=bool)
        self.pool = self.opp_net = None
        self.elo = Elo()
        if cfg.selfplay:
            for i, m in enumerate(env.meta):
                if m.policy == cfg.selfplay:
                    learner[i] = m.team == (m.arena % 2)
            self.pool = OpponentPool(seed=cfg.seed)
            self.pool.add("snap_0", self.nets[cfg.selfplay].state_dict())
            self.opp_net = clone_module(self.nets[cfg.selfplay]).eval()
            self.opp_name = "snap_0"
            self.elo.r["snap_0"] = self.elo.get("learner")
        self.learner = learner
        self.groups = {p: np.where((pol == p) & learner)[0] for p in env.policies}
        self.groups = {p: g for p, g in self.groups.items() if len(g)}
        self.opp_idx = np.where(~learner)[0]
        self.n_learn = int(learner.sum())
        self.rms = {p: RunningMeanStd() for p in self.groups}
        self.ret = np.zeros(self.N)
        self.stack = np.zeros((self.N, self.K, H, W, 3), dtype=np.uint8)
        from torch.utils.tensorboard import SummaryWriter

        self.tb = SummaryWriter(str(self.run_dir / "tb"))
        self.ep_log = open(self.run_dir / "episodes.jsonl", "a")
        self.step = 0
        print(f"[ppo] run={self.run_dir.name} device={self.device} agents={self.N} learners={self.n_learn} "
              f"policies={ {p: len(g) for p, g in self.groups.items()} } opponents={len(self.opp_idx)}", flush=True)

    # ------------------------------------------------------------------ helpers
    def _stacked(self, idx=None) -> np.ndarray:
        s = self.stack if idx is None else self.stack[idx]
        n = s.shape[0]
        return s.transpose(0, 1, 4, 2, 3).reshape(n, self.K * 3, s.shape[2], s.shape[3])

    def _push_frames(self, pov: np.ndarray, done: np.ndarray) -> None:
        self.stack[:, :-1] = self.stack[:, 1:].copy()
        self.stack[:, -1] = pov
        if done.any():
            self.stack[done] = pov[done][:, None]

    def _log_episodes(self, infos: list[dict]) -> list[dict]:
        learn_infos = []
        for d in infos:
            i = int(d["agent"])
            if not self.learner[i]:
                continue
            d = dict(d)
            d["policy"] = self.env.meta[i].policy
            learn_infos.append(d)
            self.ep_log.write(json.dumps({"step": self.step, **d}) + "\n")
            if self.pool is not None and d["policy"] == self.cfg.selfplay and "result" in d:
                self.elo.update("learner", self.opp_name, (float(d["result"]) + 1.0) / 2.0)
        return learn_infos

    def random_baseline(self, decisions: int) -> dict:
        rng = np.random.default_rng(self.cfg.seed + 99)
        self.env.reset(seed=self.cfg.seed + 99)
        eps = defaultdict(list)
        for _ in range(decisions):
            obs = self.env.step(self.env.random_actions(rng))
            for d in obs.infos:
                pol = self.env.meta[int(d["agent"])].policy
                for k in LOG_KEYS:
                    if k in d and isinstance(d[k], (int, float)):
                        eps[f"{pol}/{k}"].append(float(d[k]))
        out = {k: float(np.mean(v)) for k, v in eps.items()}
        out["episodes"] = len(eps.get(next(iter(eps), ""), []))
        (self.run_dir / "baseline_random.json").write_text(json.dumps(out, indent=1))
        print("[ppo] random baseline:", json.dumps({k: round(v, 3) for k, v in out.items()}), flush=True)
        return out

    # ------------------------------------------------------------------ main loop
    def train(self) -> Path:
        cfg, env, dev = self.cfg, self.env, self.device
        if cfg.baseline_steps > 0:
            self.random_baseline(cfg.baseline_steps)
        T, K = cfg.rollout, self.K
        H, W, _ = env.pov_shape
        bufs = {}
        for p, g in self.groups.items():
            n = len(g)
            bufs[p] = {
                "pov": torch.zeros((T, n, 3 * K, H, W), dtype=torch.uint8),
                "vec": torch.zeros((T, n, env.vec_dim)),
                "act": torch.zeros((T, n, len(env.nvec)), dtype=torch.long),
                "logp": torch.zeros((T, n)), "val": torch.zeros((T, n)),
                "rew": torch.zeros((T, n)), "done": torch.zeros((T, n)),
            }
        obs = env.reset(seed=cfg.seed)
        self.stack[:] = obs.pov[:, None]
        vec = obs.vec.copy()
        n_updates = max(1, cfg.total_steps // (T * self.n_learn))
        t_start = time.perf_counter()
        for update in range(1, n_updates + 1):
            frac = 1.0 - (update - 1) / n_updates
            for p, opt in self.opts.items():
                opt.param_groups[0]["lr"] = cfg.lr * (frac if cfg.anneal_lr else 1.0)
            ep_infos: list[dict] = []
            t_roll = time.perf_counter()
            for t in range(T):
                stacked = self._stacked()
                pov_t = torch.from_numpy(stacked).to(dev)
                vec_t = torch.from_numpy(vec).to(dev)
                actions = np.zeros((self.N, len(env.nvec)), dtype=np.int32)
                with torch.no_grad():
                    for p, g in self.groups.items():
                        gi = torch.from_numpy(g).to(dev)
                        a, logp, _, v = self.nets[p].act(pov_t[gi], vec_t[gi])
                        b = bufs[p]
                        b["pov"][t] = torch.from_numpy(stacked[g])
                        b["vec"][t] = torch.from_numpy(vec[g])
                        b["act"][t] = a.cpu()
                        b["logp"][t] = logp.cpu()
                        b["val"][t] = v.cpu()
                        actions[g] = a.cpu().numpy()
                    if len(self.opp_idx):
                        oi = torch.from_numpy(self.opp_idx).to(dev)
                        a, _, _, _ = self.opp_net.act(pov_t[oi], vec_t[oi])
                        actions[self.opp_idx] = a.cpu().numpy()
                obs = env.step(actions)
                r = obs.reward.astype(np.float64)
                self.ret = self.ret * cfg.gamma + r
                for p, g in self.groups.items():
                    rn = r[g]
                    if cfg.norm_reward:
                        self.rms[p].update(self.ret[g])
                        rn = np.clip(r[g] / np.sqrt(self.rms[p].var + 1e-8), -10, 10)
                    bufs[p]["rew"][t] = torch.from_numpy(rn.astype(np.float32))
                    bufs[p]["done"][t] = torch.from_numpy(obs.done[g].astype(np.float32))
                self.ret[obs.done] = 0.0
                self._push_frames(obs.pov, obs.done)
                vec = obs.vec.copy()
                self.step += self.n_learn
                if obs.infos:
                    li = self._log_episodes(obs.infos)
                    ep_infos += li
                    if self.curr is not None and li:
                        newp = self.curr.update(li)
                        if newp is not None:
                            env.set_params(newp)
                            print(f"[ppo] curriculum -> level {self.curr.level} {newp}", flush=True)
            roll_time = time.perf_counter() - t_roll
            # ---------------------------------------------------------- learn
            t_learn = time.perf_counter()
            stacked = self._stacked()
            stats = {}
            for p, g in self.groups.items():
                b = bufs[p]
                with torch.no_grad():
                    gi = torch.from_numpy(g)
                    next_v = self.nets[p].value(torch.from_numpy(stacked[g]).to(dev), torch.from_numpy(vec[g]).to(dev)).cpu()
                adv = torch.zeros((T, len(g)))
                last = torch.zeros(len(g))
                for s in reversed(range(T)):
                    nv = next_v if s == T - 1 else b["val"][s + 1]
                    nonterm = 1.0 - b["done"][s]
                    delta = b["rew"][s] + cfg.gamma * nv * nonterm - b["val"][s]
                    last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                    adv[s] = last
                rets = adv + b["val"]
                stats[p] = self._update(p, b, adv, rets)
            learn_time = time.perf_counter() - t_learn
            # ---------------------------------------------------------- bookkeeping
            sps = int(self.step / (time.perf_counter() - t_start))
            self.tb.add_scalar("perf/sps", sps, self.step)
            self.tb.add_scalar("perf/rollout_s", roll_time, self.step)
            self.tb.add_scalar("perf/learn_s", learn_time, self.step)
            for p, st in stats.items():
                for k, v in st.items():
                    self.tb.add_scalar(f"{p}/{k}", v, self.step)
            summary = defaultdict(list)
            for d in ep_infos:
                for k in LOG_KEYS:
                    if k in d and isinstance(d[k], (int, float)):
                        summary[f"{d['policy']}/{k}"].append(float(d[k]))
            for k, v in summary.items():
                self.tb.add_scalar(f"episode/{k}", float(np.mean(v)), self.step)
            if self.curr is not None:
                self.tb.add_scalar("curriculum/level", self.curr.level, self.step)
            if self.pool is not None:
                self.tb.add_scalar("selfplay/elo_learner", self.elo.get("learner"), self.step)
                if update % cfg.snapshot_every == 0:
                    sp = cfg.selfplay
                    snap = f"snap_{update}"
                    self.pool.add(snap, self.nets[sp].state_dict())
                    self.elo.r[snap] = self.elo.get("learner")
                    self.opp_name, sd = self.pool.sample()
                    self.opp_net.load_state_dict(sd)
            msg = " ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(summary.items()) if k.split("/")[1] in
                           ("ep_return", "win", "kills", "platforms", "hidden_frac", "tags", "stage", "difficulty"))
            elo = f" elo={self.elo.get('learner'):.0f} vs {self.opp_name}" if self.pool is not None else ""
            print(f"[ppo] upd {update}/{n_updates} step={self.step} sps={sps} eps={len(ep_infos)} {msg}{elo} "
                  f"(roll {roll_time:.1f}s learn {learn_time:.1f}s)", flush=True)
            if update % cfg.pov_png_every == 0 or update == 1:
                save_pov_grid(obs.pov[: min(16, self.N)], self.run_dir / "povs" / f"update_{update:04d}.png",
                              title=f"{cfg.scenario} update {update} step {self.step}")
            if update % cfg.save_every == 0 or update == n_updates:
                self.save()
        self.save()
        self.close()
        return self.run_dir

    def _update(self, p: str, b: dict, adv: torch.Tensor, rets: torch.Tensor) -> dict:
        cfg, dev, net, opt = self.cfg, self.device, self.nets[p], self.opts[p]
        T, n = adv.shape
        flat = lambda x: x.reshape(T * n, *x.shape[2:])  # noqa: E731
        pov, vec, act = flat(b["pov"]), flat(b["vec"]), flat(b["act"])
        old_logp, old_val, A, R = flat(b["logp"]), flat(b["val"]), flat(adv), flat(rets)
        B = T * n
        mb = max(1, B // cfg.minibatches)
        clipfracs, kls = [], []
        for _ in range(cfg.epochs):
            perm = torch.randperm(B)
            for s in range(0, B, mb):
                i = perm[s : s + mb]
                _, logp, ent, v = net.act(pov[i].to(dev), vec[i].to(dev), act[i].to(dev))
                ratio = (logp - old_logp[i].to(dev)).exp()
                a = A[i].to(dev)
                a = (a - a.mean()) / (a.std() + 1e-8)
                pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * ((v - R[i].to(dev)) ** 2).mean()
                loss = pg - cfg.ent_coef * ent.mean() + cfg.vf_coef * v_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(((ratio - 1) - ratio.log()).mean().item())
                    clipfracs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
        var_y = R.var().item()
        ev = 1 - (R - old_val).var().item() / var_y if var_y > 0 else float("nan")
        return {"loss/policy": pg.item(), "loss/value": v_loss.item(), "loss/entropy": ent.mean().item(),
                "loss/approx_kl": float(np.mean(kls)), "loss/clipfrac": float(np.mean(clipfracs)),
                "loss/explained_var": ev, "lr": opt.param_groups[0]["lr"]}

    def save(self) -> Path:
        path = self.run_dir / "latest.pt"
        torch.save({
            "cfg": asdict(self.cfg), "step": self.step,
            "policies": {p: n.state_dict() for p, n in self.nets.items()},
            "nvec": self.env.nvec, "vec_dim": self.env.vec_dim, "pov_shape": self.env.pov_shape,
            "elo": dict(self.elo.r), "curriculum_level": None if self.curr is None else self.curr.level,
        }, path)
        return path

    def close(self) -> None:
        self.env.close()
        self.tb.close()
        self.ep_log.close()
