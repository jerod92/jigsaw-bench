"""Run an AI model against the jigsaw environment and score it."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import numpy as np

from .environment import ActionPoint, JigsawEnvironment


class JigsawModel(Protocol):
    """Any callable that maps a rendered observation to an action dict."""

    def __call__(self, observation: np.ndarray, step: int) -> dict[int, ActionPoint]: ...


@dataclass
class BenchmarkResult:
    solved: bool
    steps: int
    wall_seconds: float
    final_position_error_px: dict[int, float] = field(default_factory=dict)
    final_rotation_error_deg: dict[int, float] = field(default_factory=dict)
    piece_correct: dict[int, bool] = field(default_factory=dict)
    piecewise_score: float = 0.0
    history: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "solved": self.solved,
            "steps": self.steps,
            "wall_seconds": round(self.wall_seconds, 3),
            "piecewise_score": round(self.piecewise_score, 4),
            "median_pos_error_px": float(np.median(list(self.final_position_error_px.values()) or [0.0])),
            "median_rot_error_deg": float(np.median(list(self.final_rotation_error_deg.values()) or [0.0])),
        }


def _piece_errors(env: JigsawEnvironment) -> tuple[dict[int, float], dict[int, float]]:
    pos_err: dict[int, float] = {}
    rot_err: dict[int, float] = {}
    centroids = env.piece_centroids()
    targets = env.target_centroids()
    rotations = env.piece_rotations()
    for idx, (cx, cy) in centroids.items():
        tx, ty = targets[idx]
        pos_err[idx] = math.hypot(cx - tx, cy - ty)
        r = ((rotations[idx] + 180) % 360) - 180
        rot_err[idx] = abs(r)
    return pos_err, rot_err


def _score(pos_err: dict[int, float], rot_err: dict[int, float], pos_tol: float, rot_tol: float) -> tuple[float, dict[int, bool]]:
    """Per-piece score: 1 if both errors within tolerance, otherwise smooth decay."""
    correct: dict[int, bool] = {}
    total = 0.0
    n = max(1, len(pos_err))
    for idx, pe in pos_err.items():
        re = rot_err[idx]
        ok = pe <= pos_tol and re <= rot_tol
        correct[idx] = ok
        # Smooth piecewise score (range 0..1): exponential falloff outside tolerance.
        s_pos = math.exp(-max(0.0, pe - pos_tol) / max(pos_tol, 1.0))
        s_rot = math.exp(-max(0.0, re - rot_tol) / max(rot_tol, 1.0))
        total += s_pos * s_rot
    return total / n, correct


def benchmark_model(
    env: JigsawEnvironment,
    model: JigsawModel,
    *,
    max_steps: int = 5_000,
    snap_to: bool = False,
    snap_pos_threshold_px: float = 20.0,
    snap_rot_threshold_deg: float = 12.0,
    pos_tol_px: float = 4.0,
    rot_tol_deg: float = 3.0,
    record_history: bool = False,
    on_step: Callable[[int, np.ndarray, dict[int, ActionPoint]], None] | None = None,
) -> BenchmarkResult:
    """Run `model` until solved or max_steps. If snap_to, attempt to snap any piece released
    near its target after every step (easy mode)."""
    obs = env.reset()
    t0 = time.perf_counter()
    history: list[dict] = []

    last_held: dict[int, int | None] = {}
    for step in range(1, max_steps + 1):
        action = model(obs, step)
        obs = env.step(action)

        if snap_to:
            currently_held = {cid: c.held_piece for cid, c in env.cursors.items()}
            for cid, prev in list(last_held.items()):
                now = currently_held.get(cid)
                if prev is not None and prev != now:
                    env.snap_piece(prev, snap_pos_threshold_px, snap_rot_threshold_deg)
            for cid, cur in currently_held.items():
                last_held[cid] = cur

        if record_history:
            pos_err, rot_err = _piece_errors(env)
            history.append({
                "step": step,
                "median_pos_error": float(np.median(list(pos_err.values()))),
                "median_rot_error": float(np.median(list(rot_err.values()))),
            })

        if on_step is not None:
            on_step(step, obs, action)

        if env.is_solved(pos_tol_px, rot_tol_deg):
            break

    wall = time.perf_counter() - t0
    pos_err, rot_err = _piece_errors(env)
    pscore, correct = _score(pos_err, rot_err, pos_tol_px, rot_tol_deg)

    return BenchmarkResult(
        solved=env.is_solved(pos_tol_px, rot_tol_deg),
        steps=step,
        wall_seconds=wall,
        final_position_error_px=pos_err,
        final_rotation_error_deg=rot_err,
        piece_correct=correct,
        piecewise_score=pscore,
        history=history,
    )
