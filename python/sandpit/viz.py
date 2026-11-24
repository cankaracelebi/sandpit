"""Visualisation helpers: POV grids, image strips, learning curves, GIFs."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return img


def pov_grid_image(povs: np.ndarray, scale: int = 3, cols: int | None = None, labels: list[str] | None = None,
                   title: str | None = None) -> Image.Image:
    n, h, w, _ = povs.shape
    cols = cols or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    pad = 4
    top = 20 if title else 0
    canvas = Image.new("RGB", (cols * (w * scale + pad) + pad, rows * (h * scale + pad) + pad + top), (24, 24, 28))
    for i in range(n):
        im = Image.fromarray(povs[i]).resize((w * scale, h * scale), Image.NEAREST)
        if labels:
            im = _label(im, labels[i])
        r, c = divmod(i, cols)
        canvas.paste(im, (pad + c * (w * scale + pad), top + pad + r * (h * scale + pad)))
    if title:
        ImageDraw.Draw(canvas).text((6, 4), title, fill=(230, 230, 230))
    return canvas


def save_pov_grid(povs: np.ndarray, path: str | Path, **kw) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pov_grid_image(np.asarray(povs), **kw).save(path)
    return path


def compose_row(paths: list[str | Path], out: str | Path, height: int = 360, title: str | None = None) -> Path:
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims = [im.resize((max(1, int(im.width * height / im.height)), height),
                     Image.NEAREST if im.height < 128 else Image.BILINEAR) for im in ims]
    top = 24 if title else 0
    W = sum(im.width for im in ims) + 6 * (len(ims) + 1)
    canvas = Image.new("RGB", (W, height + 12 + top), (24, 24, 28))
    x = 6
    for im in ims:
        canvas.paste(im, (x, 6 + top))
        x += im.width + 6
    if title:
        ImageDraw.Draw(canvas).text((8, 6), title, fill=(235, 235, 235))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 15, scale: int = 1) -> Path:
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if scale != 1:
        frames = [np.asarray(Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST))
                  for f in frames]
    iio.imwrite(path, np.stack(frames), duration=1000 / fps, loop=0)
    return path


def plot_curves(series: dict[str, tuple[np.ndarray, np.ndarray]], path: str | Path, title: str = "",
                xlabel: str = "env steps", ylabel: str = "", smooth: int = 1, baseline: float | None = None,
                baseline_label: str = "random") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=120)
    for name, (x, y) in series.items():
        x, y = np.asarray(x, float), np.asarray(y, float)
        if smooth > 1 and len(y) >= smooth:
            k = np.ones(smooth) / smooth
            y = np.convolve(y, k, mode="valid")
            x = x[smooth - 1 :]
        ax.plot(x, y, label=name, linewidth=1.8)
    if baseline is not None:
        ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if series or baseline is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
