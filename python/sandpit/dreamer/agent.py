"""World model + imagination actor-critic (DreamerV3-style) with compact default sizes."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .nets import RSSM, Decoder, Encoder, MultiCategoricalActor, TwoHot, mlp, onehot_actions, symlog


@dataclass
class DreamerConfig:
    deter: int = 256
    stoch: int = 16
    classes: int = 16
    hidden: int = 256
    depth: int = 32
    wm_lr: float = 3e-4
    ac_lr: float = 8e-5
    horizon: int = 15
    gamma: float = 0.985
    lam: float = 0.95
    ent_coef: float = 3e-4
    kl_free: float = 1.0
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    slow_critic_decay: float = 0.98
    grad_clip: float = 100.0


class Dreamer(nn.Module):
    def __init__(self, nvec: tuple[int, ...], vec_dim: int, cfg: DreamerConfig):
        super().__init__()
        self.cfg, self.nvec, self.vec_dim = cfg, tuple(nvec), vec_dim
        self.act_dim = sum(self.nvec)
        self.enc = Encoder(vec_dim, cfg.depth)
        self.rssm = RSSM(self.enc.out_dim, self.act_dim, cfg.deter, cfg.stoch, cfg.classes, cfg.hidden)
        F = self.rssm.feat_dim
        self.dec = Decoder(F, vec_dim, cfg.depth, cfg.hidden)
        self.rew_head = mlp(F, 255, cfg.hidden, out_zero=True)
        self.cont_head = mlp(F, 1, cfg.hidden)
        self.actor = MultiCategoricalActor(F, self.nvec, cfg.hidden)
        self.critic = mlp(F, 255, cfg.hidden, out_zero=True)
        self.slow_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.twohot = TwoHot()
        wm_params = [*self.enc.parameters(), *self.rssm.parameters(), *self.dec.parameters(),
                     *self.rew_head.parameters(), *self.cont_head.parameters()]
        self.wm_opt = torch.optim.Adam(wm_params, lr=cfg.wm_lr, eps=1e-8)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.ac_lr, eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.ac_lr, eps=1e-5)
        self.wm_params = wm_params
        self.register_buffer("ret_scale", torch.ones(()))

    # ------------------------------------------------------------------ helpers
    def _to(self, batch: dict, dev) -> dict:
        return {k: torch.as_tensor(v, device=dev) for k, v in batch.items()}

    def prev_actions(self, act: torch.Tensor, first: torch.Tensor) -> torch.Tensor:
        oh = onehot_actions(act, self.nvec)
        prev = torch.cat([torch.zeros_like(oh[:, :1]), oh[:, :-1]], 1)
        return prev * (1.0 - first.float().unsqueeze(-1))

    # ------------------------------------------------------------------ world model
    def train_wm(self, batch: dict, dev) -> tuple[dict, dict]:
        b = self._to(batch, dev)
        embed = self.enc(b["pov"], b["vec"])
        states, post_l, prior_l, _ = self.rssm.observe(embed, self.prev_actions(b["act"], b["first"]), b["first"])
        feat = RSSM.feat(states)
        img, vec_hat = self.dec(feat)
        target = b["pov"].float() / 255.0 - 0.5
        rec_img = ((img - target) ** 2).sum((-1, -2, -3))
        rec_vec = ((vec_hat - symlog(b["vec"])) ** 2).sum(-1)
        rew_loss = self.twohot.loss(self.rew_head(feat), b["rew"])
        cont_t = 1.0 - b["term"].float()
        cont_loss = nn.functional.binary_cross_entropy_with_logits(self.cont_head(feat).squeeze(-1), cont_t, reduction="none")
        dyn = self.rssm.kl(post_l.detach(), prior_l).clamp(min=self.cfg.kl_free)
        rep = self.rssm.kl(post_l, prior_l.detach()).clamp(min=self.cfg.kl_free)
        loss = (rec_img + rec_vec + rew_loss + cont_loss + self.cfg.dyn_scale * dyn + self.cfg.rep_scale * rep).mean()
        self.wm_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm_params, 1000.0)
        self.wm_opt.step()
        metrics = {"wm/loss": loss.item(), "wm/rec_img": rec_img.mean().item(), "wm/rec_vec": rec_vec.mean().item(),
                   "wm/rew": rew_loss.mean().item(), "wm/cont": cont_loss.mean().item(),
                   "wm/kl": self.rssm.kl(post_l, prior_l).mean().item()}
        starts = {k: v.detach().reshape(-1, v.shape[-1]) for k, v in states.items()}
        return metrics, starts

    # ------------------------------------------------------------------ behaviour
    def train_ac(self, starts: dict) -> dict:
        cfg = self.cfg
        state = starts
        feats, logps, ents = [RSSM.feat(state)], [], []
        self.rssm.requires_grad_(False)
        for _ in range(cfg.horizon):
            a, logp, ent = self.actor.sample(feats[-1].detach())
            state, _ = self.rssm.img_step(state, onehot_actions(a, self.nvec))
            feats.append(RSSM.feat(state))
            logps.append(logp)
            ents.append(ent)
        self.rssm.requires_grad_(True)
        feat = torch.stack(feats, 0).detach()                       # (H+1, N, F)
        with torch.no_grad():
            rew = self.twohot.mean(self.rew_head(feat))[1:]          # reward on arriving at t+1
            cont = torch.sigmoid(self.cont_head(feat)).squeeze(-1)[1:]
            val = self.twohot.mean(self.slow_critic(feat))
            disc = cfg.gamma * cont
            rets = [val[-1]]
            for t in reversed(range(cfg.horizon)):
                rets.append(rew[t] + disc[t] * ((1 - cfg.lam) * val[t + 1] + cfg.lam * rets[-1]))
            ret = torch.stack(rets[::-1][:-1], 0)                    # (H, N) lambda-returns for t=0..H-1
            weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0)
            q = torch.quantile(ret.flatten().float().cpu(), torch.tensor([0.05, 0.95]))
            lo, hi = q[0].to(ret.device), q[1].to(ret.device)
            self.ret_scale.mul_(0.99).add_(0.01 * (hi - lo))
            scale = torch.clamp(self.ret_scale, min=1.0)
            adv = (ret - val[:-1]) / scale
        logp = torch.stack(logps, 0)
        ent = torch.stack(ents, 0)
        actor_loss = -(weight * (logp * adv + cfg.ent_coef * ent)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
        self.actor_opt.step()
        v_logits = self.critic(feat[:-1])
        slow_target = self.twohot.mean(self.slow_critic(feat[:-1])).detach()
        critic_loss = (weight * (self.twohot.loss(v_logits, ret) + self.twohot.loss(v_logits, slow_target))).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()
        with torch.no_grad():
            for p, sp in zip(self.critic.parameters(), self.slow_critic.parameters()):
                sp.mul_(cfg.slow_critic_decay).add_((1 - cfg.slow_critic_decay) * p)
        return {"ac/actor_loss": actor_loss.item(), "ac/critic_loss": critic_loss.item(), "ac/entropy": ent.mean().item(),
                "ac/imag_return": ret.mean().item(), "ac/imag_reward": rew.mean().item(), "ac/ret_scale": float(self.ret_scale)}

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def policy_step(self, state: dict | None, prev_act: torch.Tensor, pov, vec, first, deterministic=False):
        embed = self.enc(pov, vec)
        if state is None:
            state = self.rssm.initial(pov.shape[0], pov.device)
        state, _, _ = self.rssm.obs_step(state, onehot_actions(prev_act, self.nvec), embed, first)
        a, _, _ = self.actor.sample(RSSM.feat(state), deterministic)
        return a, state

    # ------------------------------------------------------------------ open-loop video prediction
    @torch.no_grad()
    def openloop(self, batch: dict, dev, context: int = 8) -> tuple[np.ndarray, np.ndarray]:
        b = self._to(batch, dev)
        L = b["pov"].shape[1]
        embed = self.enc(b["pov"][:, :context], b["vec"][:, :context])
        prev = self.prev_actions(b["act"], b["first"])
        states, _, _, last = self.rssm.observe(embed, prev[:, :context], b["first"][:, :context])
        recon, _ = self.dec(RSSM.feat(states))
        state, preds = last, []
        for t in range(context, L):
            state, _ = self.rssm.img_step(state, prev[:, t])
            img, _ = self.dec(RSSM.feat(state))
            preds.append(img)
        pred = torch.cat([recon, torch.stack(preds, 1)], 1)
        pred = ((pred + 0.5).clamp(0, 1) * 255).byte().cpu().numpy()
        return b["pov"].cpu().numpy(), pred
