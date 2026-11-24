import numpy as np
import torch

from sandpit.dreamer.agent import Dreamer, DreamerConfig
from sandpit.dreamer.nets import TwoHot, symexp, symlog
from sandpit.dreamer.replay import StreamReplay, StreamWriter

NVEC = (3, 3, 5, 3, 2, 2)


def test_symlog_roundtrip():
    x = torch.tensor([-100.0, -1.0, 0.0, 0.5, 42.0])
    assert torch.allclose(symexp(symlog(x)), x, atol=1e-4)


def test_twohot_mean_matches_target():
    th = TwoHot()
    target = torch.tensor([0.0, 1.0, -3.5, 12.0])
    logits = torch.zeros(4, 255, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.5)
    for _ in range(300):
        loss = th.loss(logits, target).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.allclose(th.mean(logits), target, atol=0.15)


def _fake_replay(n=4, cap=80, vec=9):
    rep = StreamReplay(n, cap, (64, 64, 3), vec, 6)
    ids = np.arange(n)
    w = StreamWriter(rep, ids)
    rng = np.random.default_rng(0)
    w.start(rng.integers(0, 255, (n, 64, 64, 3), dtype=np.uint8), rng.normal(size=(n, vec)).astype(np.float32))
    for t in range(60):
        done = rng.random(n) < 0.05
        w.step(rng.integers(0, 2, (n, 6)), rng.integers(0, 255, (n, 64, 64, 3), dtype=np.uint8),
               rng.normal(size=(n, vec)).astype(np.float32), rng.normal(size=n).astype(np.float32), done, np.zeros(n, bool))
    return rep


def test_replay_sample_shapes_and_first_flag():
    rep = _fake_replay()
    b = rep.sample(3, 16)
    assert b["pov"].shape == (3, 16, 64, 64, 3) and b["pov"].dtype == np.uint8
    assert b["act"].shape == (3, 16, 6) and b["first"][:, 0].all()


def test_dreamer_train_step_and_openloop_shapes():
    torch.manual_seed(0)
    cfg = DreamerConfig(deter=64, stoch=4, classes=4, hidden=64, depth=8, horizon=5)
    rep = _fake_replay()
    agent = Dreamer(NVEC, 9, cfg)
    m, starts = agent.train_wm(rep.sample(2, 12), "cpu")
    assert np.isfinite(list(m.values())).all()
    assert starts["deter"].shape == (24, 64)
    m2 = agent.train_ac(starts)
    assert np.isfinite(list(m2.values())).all()
    gt, pred = agent.openloop(rep.sample(2, 12), "cpu", context=4)
    assert gt.shape == pred.shape == (2, 12, 64, 64, 3)
    a, state = agent.policy_step(None, torch.zeros(5, 6, dtype=torch.long), torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
                                 torch.zeros(5, 9), torch.ones(5, dtype=torch.bool))
    assert a.shape == (5, 6) and (a < torch.tensor(NVEC)).all()
