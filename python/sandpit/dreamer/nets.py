"""DreamerV3-style building blocks (compact PyTorch re-implementation for 64x64 RGB + vector obs).

References: Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3).
Simplifications relative to the paper: nn.GRUCell instead of the block-diagonal LN-GRU,
Adam instead of LaProp, smaller default sizes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class TwoHot:
    """Discrete regression in symlog space with two-hot targets (rewards & values)."""

    def __init__(self, bins: int = 255, low: float = -20.0, high: float = 20.0):
        self.bins = torch.linspace(low, high, bins)

    def _b(self, ref: torch.Tensor) -> torch.Tensor:
        if self.bins.device != ref.device:
            self.bins = self.bins.to(ref.device)
        return self.bins

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        b = self._b(logits)
        return symexp((logits.softmax(-1) * b).sum(-1))

    def loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b = self._b(logits)
        y = symlog(target).clamp(b[0], b[-1])
        idx = torch.searchsorted(b, y.contiguous(), right=True).clamp(1, len(b) - 1)
        lo, hi = b[idx - 1], b[idx]
        w_hi = ((y - lo) / (hi - lo)).clamp(0, 1)
        logp = logits.log_softmax(-1)
        lp_lo = logp.gather(-1, (idx - 1).unsqueeze(-1)).squeeze(-1)
        lp_hi = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        return -((1 - w_hi) * lp_lo + w_hi * lp_hi)


def mlp(inp: int, out: int, hidden: int, layers: int = 2, out_zero: bool = False) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
        d = hidden
    last = nn.Linear(d, out)
    if out_zero:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
    mods.append(last)
    return nn.Sequential(*mods)


class Encoder(nn.Module):
    """64x64x3 uint8 + vec -> embedding."""

    def __init__(self, vec_dim: int, depth: int = 32, vec_hidden: int = 128):
        super().__init__()
        ch = [3, depth, 2 * depth, 4 * depth, 8 * depth]
        mods: list[nn.Module] = []
        for i in range(4):
            mods += [nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1), nn.GroupNorm(1, ch[i + 1]), nn.SiLU()]
        self.cnn = nn.Sequential(*mods, nn.Flatten())
        self.vec = mlp(vec_dim, vec_hidden, vec_hidden, layers=1)
        self.out_dim = 8 * depth * 4 * 4 + vec_hidden

    def forward(self, pov: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        # pov: (..., H, W, 3) uint8 ; vec: (..., V)
        lead = pov.shape[:-3]
        x = pov.reshape(-1, *pov.shape[-3:]).permute(0, 3, 1, 2).float() / 255.0 - 0.5
        e = self.cnn(x)
        v = self.vec(symlog(vec.reshape(-1, vec.shape[-1])))
        return torch.cat([e, v], -1).reshape(*lead, -1)


class Decoder(nn.Module):
    """feature -> 64x64x3 image (mean, in [-0.5, 0.5] space) + symlog vec."""

    def __init__(self, feat_dim: int, vec_dim: int, depth: int = 32, hidden: int = 256):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(feat_dim, 8 * depth * 4 * 4)
        ch = [8 * depth, 4 * depth, 2 * depth, depth]
        mods: list[nn.Module] = []
        for i in range(3):
            mods += [nn.ConvTranspose2d(ch[i], ch[i + 1], 4, 2, 1), nn.GroupNorm(1, ch[i + 1]), nn.SiLU()]
        mods.append(nn.ConvTranspose2d(depth, 3, 4, 2, 1))
        self.deconv = nn.Sequential(*mods)
        self.vec = mlp(feat_dim, vec_dim, hidden, layers=2)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lead = feat.shape[:-1]
        f = feat.reshape(-1, feat.shape[-1])
        x = self.fc(f).reshape(-1, 8 * self.depth, 4, 4)
        img = self.deconv(x).permute(0, 2, 3, 1)  # (N, 64, 64, 3)
        return img.reshape(*lead, *img.shape[1:]), self.vec(f).reshape(*lead, -1)


class RSSM(nn.Module):
    """Recurrent state-space model with discrete (categorical) stochastic latents."""

    def __init__(self, embed_dim: int, act_dim: int, deter: int = 256, stoch: int = 16, classes: int = 16,
                 hidden: int = 256, unimix: float = 0.01):
        super().__init__()
        self.deter, self.stoch, self.classes, self.unimix = deter, stoch, classes, unimix
        self.img_in = nn.Sequential(nn.Linear(stoch * classes + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter)
        self.prior = mlp(deter, stoch * classes, hidden, layers=1)
        self.post = mlp(deter + embed_dim, stoch * classes, hidden, layers=1)
        self.feat_dim = deter + stoch * classes

    def initial(self, n: int, device) -> dict:
        return {"deter": torch.zeros(n, self.deter, device=device),
                "stoch": torch.zeros(n, self.stoch * self.classes, device=device)}

    def _probs(self, logits: torch.Tensor) -> torch.Tensor:
        p = logits.reshape(*logits.shape[:-1], self.stoch, self.classes).softmax(-1)
        return (1 - self.unimix) * p + self.unimix / self.classes

    def _sample(self, probs: torch.Tensor) -> torch.Tensor:
        idx = torch.distributions.Categorical(probs=probs).sample()
        onehot = F.one_hot(idx, self.classes).to(probs.dtype)
        st = onehot + probs - probs.detach()  # straight-through gradients
        return st.reshape(*st.shape[:-2], self.stoch * self.classes)

    @staticmethod
    def feat(state: dict) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], -1)

    def img_step(self, state: dict, action: torch.Tensor) -> tuple[dict, torch.Tensor]:
        x = self.img_in(torch.cat([state["stoch"], action], -1))
        deter = self.gru(x, state["deter"])
        logits = self.prior(deter)
        stoch = self._sample(self._probs(logits))
        return {"deter": deter, "stoch": stoch}, logits

    def obs_step(self, state: dict, prev_action: torch.Tensor, embed: torch.Tensor, is_first: torch.Tensor):
        keep = (1.0 - is_first.float()).unsqueeze(-1)
        state = {k: v * keep for k, v in state.items()}
        prev_action = prev_action * keep
        prior_state, prior_logits = self.img_step(state, prev_action)
        post_logits = self.post(torch.cat([prior_state["deter"], embed], -1))
        stoch = self._sample(self._probs(post_logits))
        return {"deter": prior_state["deter"], "stoch": stoch}, post_logits, prior_logits

    def observe(self, embeds: torch.Tensor, prev_actions: torch.Tensor, is_first: torch.Tensor, state: dict | None = None):
        B, L = embeds.shape[:2]
        state = state or self.initial(B, embeds.device)
        deters, stochs, posts, priors = [], [], [], []
        for t in range(L):
            state, post_l, prior_l = self.obs_step(state, prev_actions[:, t], embeds[:, t], is_first[:, t])
            deters.append(state["deter"])
            stochs.append(state["stoch"])
            posts.append(post_l)
            priors.append(prior_l)
        st = {"deter": torch.stack(deters, 1), "stoch": torch.stack(stochs, 1)}
        return st, torch.stack(posts, 1), torch.stack(priors, 1), state

    def kl(self, lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
        p = self._probs(lhs_logits)
        q = self._probs(rhs_logits)
        return (p * (p.log() - q.log())).sum(-1).sum(-1)


class MultiCategoricalActor(nn.Module):
    def __init__(self, feat_dim: int, nvec: tuple[int, ...], hidden: int = 256, unimix: float = 0.01):
        super().__init__()
        self.nvec = tuple(nvec)
        self.net = mlp(feat_dim, sum(self.nvec), hidden, layers=2)
        self.unimix = unimix

    def dists(self, feat: torch.Tensor):
        out = []
        for l, n in zip(torch.split(self.net(feat), self.nvec, -1), self.nvec):
            p = (1 - self.unimix) * l.softmax(-1) + self.unimix / n
            out.append(torch.distributions.Categorical(probs=p))
        return out

    def sample(self, feat: torch.Tensor, deterministic: bool = False):
        ds = self.dists(feat)
        a = torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in ds], -1)
        logp = torch.stack([d.log_prob(x) for d, x in zip(ds, a.unbind(-1))], -1).sum(-1)
        ent = torch.stack([d.entropy() for d in ds], -1).sum(-1)
        return a, logp, ent


def onehot_actions(a: torch.Tensor, nvec: tuple[int, ...]) -> torch.Tensor:
    return torch.cat([F.one_hot(a[..., i].long(), n).float() for i, n in enumerate(nvec)], -1)
