"""GIF recording utilities for jigsaw benchmark rollouts.

Usage
-----
    from jigsaw_bench import record_rollout, save_gif

    frames, result = record_rollout(env, model, max_steps=2000, capture_every=8)
    save_gif(frames, "run.gif", fps=12, max_width=640)
    print(result.summary())
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .benchmark import BenchmarkResult, benchmark_model
from .environment import ActionPoint, JigsawEnvironment


def record_rollout(
    env: JigsawEnvironment,
    model: Callable,
    *,
    max_steps: int = 2000,
    capture_every: int = 8,
    snap_to: bool = True,
    snap_pos_threshold_px: float = 18.0,
    snap_rot_threshold_deg: float = 10.0,
    pos_tol_px: float = 4.0,
    rot_tol_deg: float = 3.0,
) -> tuple[list[np.ndarray], BenchmarkResult]:
    """Run *model* against *env*, capturing a frame every *capture_every* steps.

    Delegates to :func:`~jigsaw_bench.benchmark_model` via its ``on_step``
    callback so all snap / scoring logic stays in one place.

    Parameters
    ----------
    capture_every:
        Record one frame every this many environment steps.

    Returns
    -------
    frames : list of (H, W, 3) uint8 arrays (rendered canvas at each capture)
    result : BenchmarkResult
    """
    frames: list[np.ndarray] = []

    def _capture(step: int, obs: np.ndarray, _action: dict[int, ActionPoint]) -> None:
        if step == 1 or step % capture_every == 0:
            frames.append(obs.copy())

    result = benchmark_model(
        env,
        model,
        max_steps=max_steps,
        snap_to=snap_to,
        snap_pos_threshold_px=snap_pos_threshold_px,
        snap_rot_threshold_deg=snap_rot_threshold_deg,
        pos_tol_px=pos_tol_px,
        rot_tol_deg=rot_tol_deg,
        on_step=_capture,
    )

    # Always include the final frame
    final = env.render()
    if not frames or not np.array_equal(frames[-1], final):
        frames.append(final)

    return frames, result


def save_gif(
    frames: list[np.ndarray],
    path: str | Path,
    *,
    fps: float = 10.0,
    max_width: int = 640,
    loop: int = 0,
) -> Path:
    """Save *frames* as an animated GIF.

    Parameters
    ----------
    fps:
        Playback speed in frames per second.
    max_width:
        Frames wider than this are scaled down proportionally.  Keeping
        the GIF small enough to open comfortably in a browser.
    loop:
        Number of times to loop (0 = forever).

    Returns
    -------
    Path to the saved GIF.
    """
    if not frames:
        raise ValueError("frames list is empty")

    path = Path(path)
    duration_ms = max(20, int(1000 / fps))

    pil_frames: list[Image.Image] = []
    for arr in frames:
        img = Image.fromarray(arr)
        w, h = img.size
        if w > max_width:
            nh = int(h * max_width / w)
            img = img.resize((max_width, nh), Image.BILINEAR)
        pil_frames.append(img.convert("RGB"))

    # Convert to P-mode (palette) for GIF — adaptive quantisation per frame
    p_frames = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in pil_frames]

    p_frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=p_frames[1:],
        loop=loop,
        duration=duration_ms,
        optimize=True,
    )
    size_kb = path.stat().st_size // 1024
    print(f"GIF saved: {path}  ({len(frames)} frames, {size_kb} KB)")
    return path
