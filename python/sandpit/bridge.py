"""Python side of the Godot RL bridge.

Python listens on a TCP port, launches a Godot process that connects back, and then the two
exchange length-prefixed messages in lockstep:

    [u32 body_len][u32 header_len][header JSON][binary payload]

Observations arrive as one binary payload: all agents' POV images (uint8 HWC), vector obs
(float32), rewards (float32), dones (uint8) and truncations (uint8), in agent order.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT / "godot"
GODOT_BIN = os.environ.get("GODOT_BIN", "/opt/homebrew/bin/godot")
NVEC = (3, 3, 5, 3, 2, 2)


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("Godot closed the connection")
        got += k
    return buf


@dataclass
class StepResult:
    pov: np.ndarray      # (n, H, W, 3) uint8
    vec: np.ndarray      # (n, V) float32
    reward: np.ndarray   # (n,) float32
    done: np.ndarray     # (n,) bool
    trunc: np.ndarray    # (n,) bool
    infos: list          # episode-end dicts with local "agent" index
    header: dict


class GodotProcess:
    """One Godot instance holding `arenas` copies of a scenario (possibly several agents each)."""

    def __init__(
        self,
        scenario: str,
        *,
        mode: str = "train",
        seed: int = 0,
        arenas: int = 1,
        action_repeat: int = 4,
        speedup: float = 1000.0,
        pov: int = 64,
        shadows: bool = False,
        params: dict | None = None,
        window: tuple[int, int] = (256, 256),
        position: tuple[int, int] | None = None,
        godot_args: tuple[str, ...] = (),
        user_args: dict | None = None,
        log_path: str | os.PathLike | None = None,
        connect_timeout: float = 90.0,
    ):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        cmd = [GODOT_BIN, "--path", str(PROJECT_DIR), "--windowed", "--resolution", f"{window[0]}x{window[1]}"]
        if position is not None:
            cmd += ["--position", f"{position[0]},{position[1]}"]
        cmd += list(godot_args)
        cmd += [
            "--",
            f"--mode={mode}",
            f"--scenario={scenario}",
            f"--port={self.port}",
            f"--seed={seed}",
            f"--arenas={arenas}",
            f"--action_repeat={action_repeat}",
            f"--speedup={speedup}",
            f"--pov={pov}",
            f"--shadows={int(shadows)}",
        ]
        if params:
            cmd.append("--params=" + urllib.parse.quote(json.dumps(params)))
        for k, v in (user_args or {}).items():
            cmd.append(f"--{k}={v}")
        self.cmd = cmd
        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        self.server.settimeout(connect_timeout)
        try:
            self.sock, _ = self.server.accept()
        except socket.timeout as e:
            self.proc.kill()
            raise RuntimeError(f"Godot did not connect within {connect_timeout}s: {' '.join(cmd)}") from e
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        hello, _ = self._recv()
        assert hello["type"] == "hello", hello
        self.info = hello["env_info"]
        self.n = int(self.info["n_agents"])
        self.pov_shape = tuple(self.info["pov_shape"])
        self.vec_dim = int(self.info["vec_dim"])
        self.nvec = tuple(self.info["nvec"])
        self.agents = self.info["agents"]
        self._closed = False

    # ------------------------------------------------------------------ framing
    def _send(self, header: dict, payload: bytes = b"") -> None:
        hb = json.dumps(header).encode()
        self.sock.sendall(struct.pack("<II", 4 + len(hb) + len(payload), len(hb)) + hb + payload)

    def _recv(self) -> tuple[dict, memoryview]:
        (body_len,) = struct.unpack("<I", _recv_exact(self.sock, 4))
        body = _recv_exact(self.sock, body_len)
        (hlen,) = struct.unpack_from("<I", body, 0)
        header = json.loads(bytes(body[4 : 4 + hlen]))
        return header, memoryview(body)[4 + hlen :]

    def _parse_obs(self, header: dict, payload: memoryview) -> StepResult:
        n, (h, w, c), v = self.n, self.pov_shape, self.vec_dim
        o = 0
        if header.get("pov_layout") == "atlas":
            rows, cols = header["atlas_grid"]
            img = np.frombuffer(payload, np.uint8, rows * h * cols * w * c, o)
            pov = img.reshape(rows, h, cols, w, c).transpose(0, 2, 1, 3, 4).reshape(rows * cols, h, w, c)[:n]
            pov = np.ascontiguousarray(pov)
            o += rows * h * cols * w * c
        else:
            pov = np.frombuffer(payload, np.uint8, n * h * w * c, o).reshape(n, h, w, c)
            o += n * h * w * c
        vec = np.frombuffer(payload, np.float32, n * v, o).reshape(n, v)
        o += 4 * n * v
        rew = np.frombuffer(payload, np.float32, n, o)
        o += 4 * n
        done = np.frombuffer(payload, np.uint8, n, o).astype(bool)
        o += n
        trunc = np.frombuffer(payload, np.uint8, n, o).astype(bool)
        o += n
        blind = np.frombuffer(payload, np.uint8, n, o).astype(bool)
        if blind.any():
            pov = pov.copy()
            pov[blind] = 0
        return StepResult(pov, vec, rew, done, trunc, header.get("infos", []), header)

    def _recv_obs(self) -> StepResult:
        header, payload = self._recv()
        assert header["type"] == "obs", header
        return self._parse_obs(header, payload)

    # ------------------------------------------------------------------ API (split send/recv so
    # several processes can simulate in parallel)
    def send_reset(self, seed: int | None = None) -> None:
        self._send({"type": "reset"} if seed is None else {"type": "reset", "seed": int(seed)})

    def send_step(self, actions: np.ndarray) -> None:
        a = np.ascontiguousarray(actions, dtype="<i4").reshape(self.n, len(NVEC))
        self._send({"type": "step"}, a.tobytes())

    def recv(self) -> StepResult:
        return self._recv_obs()

    def reset(self, seed: int | None = None) -> StepResult:
        self.send_reset(seed)
        return self.recv()

    def step(self, actions: np.ndarray) -> StepResult:
        self.send_step(actions)
        return self.recv()

    def call(self, method: str, *args):
        self._send({"type": "call", "method": method, "args": list(args)})
        header, _ = self._recv()
        assert header["type"] == "ack", header
        return header.get("result")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send({"type": "close"})
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        for s in (getattr(self, "sock", None), self.server):
            try:
                s.close()
            except Exception:
                pass
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
