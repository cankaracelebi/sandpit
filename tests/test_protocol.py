"""Round-trip through a real Godot process (spike scenario): handshake, reset, step, call,
same-step auto-reset, and frame freshness (image reflects the decision tick's transforms)."""
import numpy as np
import pytest

from sandpit.bridge import GodotProcess


@pytest.fixture(scope="module")
def proc():
    p = GodotProcess("spike", arenas=2, params={"agents_per_arena": 2}, user_args={"minimize": 1})
    yield p
    p.close()


def test_handshake(proc):
    assert proc.n == 4 and proc.pov_shape == (64, 64, 3) and proc.nvec == (3, 3, 5, 3, 2, 2)
    assert proc.call("ping") == "pong"


def test_reset_step_and_freshness(proc):
    r = proc.reset(seed=3)
    assert r.pov.shape == (4, 64, 64, 3) and r.pov.dtype == np.uint8 and r.pov.std() > 5
    rng = np.random.default_rng(0)
    checks = mism = dones = 0
    for _ in range(120):  # spike episodes are 300 ticks = 75 decisions -> resets happen
        r = proc.step(np.stack([rng.integers(0, n, size=proc.n) for n in proc.nvec], 1))
        px = r.pov[:, 32, 32].astype(float)
        vis = (np.abs(px[:, 1] - 40) < 12) & (np.abs(px[:, 2] - 200) < 12)
        level = np.clip(np.round((px[:, 0] - 16) / 32), 0, 7)
        checks += vis.sum()
        mism += ((level != r.vec[:, -1]) & vis).sum()
        dones += r.done.sum()
        for d in r.infos:
            assert {"ep_return", "ep_len", "agent"} <= set(d)
    assert checks > 100 and mism == 0
    assert dones >= 4  # every agent's episode ended at least once (auto-reset)


def test_set_params_call(proc):
    assert proc.call("set_params", {"foo": 1}, None) is True
