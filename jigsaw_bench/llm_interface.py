"""Vanilla-LLM (computer-use style) interface around the JigsawEnvironment.

Exposes a small set of tools — move, grab, release, rotate, finished — that mutate a
single cursor position. The cursor is drawn on top of every rendered frame so a
multimodal LLM can see it. Easy-mode snap is on by default so the LLM doesn't need
sub-pixel motor control to win.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .benchmark import BenchmarkResult, _piece_errors, _score
from .environment import ActionPoint, JigsawEnvironment


TOOL_SCHEMA = {
    "move": {
        "description": "Move the cursor by (dx, dy) pixels relative to its current position. "
                       "If the cursor is currently grabbing a piece, the piece moves with it.",
        "params": {"dx": "float, pixels", "dy": "float, pixels"},
    },
    "grab": {
        "description": "Press the grab button at the cursor's current position. If a piece is under "
                       "the cursor, it becomes grabbed and moves with the cursor until release.",
        "params": {},
    },
    "release": {
        "description": "Release any currently grabbed piece. With easy-mode snap-to enabled, the "
                       "piece will jump into its target slot if released within threshold.",
        "params": {},
    },
    "rotate": {
        "description": "Rotate the currently grabbed piece by `degrees` degrees about the grab point.",
        "params": {"degrees": "float, -180..+180"},
    },
    "finished": {
        "description": "Declare the puzzle finished and end the session.",
        "params": {},
    },
}


class LLMAgent(Protocol):
    """Multimodal LLM agent. Given a frame (with cursor drawn) and a tool schema,
    return one tool call as ``(name, kwargs)``."""

    def __call__(self, frame: np.ndarray, step: int, tool_schema: dict) -> tuple[str, dict]: ...


@dataclass
class CursorState:
    x: float
    y: float
    grabbing: bool = False


class LLMCursorInterface:
    """Single-cursor wrapper around JigsawEnvironment with relative-motion controls."""

    def __init__(
        self,
        env: JigsawEnvironment,
        *,
        cursor_id: int = 0,
        snap_to: bool = True,
        snap_pos_threshold_px: float = 28.0,
        snap_rot_threshold_deg: float = 20.0,
        cursor_color: tuple[int, int, int] = (255, 64, 64),
    ):
        """Default snap thresholds: 28 px / 20 deg. Pass ``snap_rot_threshold_deg=180`` for
        position-only snap (auto-zeroes rotation regardless of pose)."""
        self.env = env
        self.cursor_id = cursor_id
        self.snap_to = snap_to
        self.snap_pos = snap_pos_threshold_px
        self.snap_rot = snap_rot_threshold_deg
        self.cursor_color = cursor_color
        self.cursor = CursorState(x=env.canvas_w / 2, y=env.canvas_h / 2)
        self._pending_rotation_deg = 0.0
        self._finished = False

    # ---- LLM-facing tools ----

    def move(self, dx: float, dy: float) -> np.ndarray:
        self.cursor.x = float(np.clip(self.cursor.x + dx, 0, self.env.canvas_w - 1))
        self.cursor.y = float(np.clip(self.cursor.y + dy, 0, self.env.canvas_h - 1))
        return self._step_env()

    def grab(self) -> np.ndarray:
        self.cursor.grabbing = True
        frame = self._step_env()
        return frame

    def release(self) -> np.ndarray:
        was_grabbing = self.cursor.grabbing
        self.cursor.grabbing = False
        held = self.env.cursors.get(self.cursor_id)
        held_idx = held.held_piece if held is not None else None
        frame = self._step_env()
        if was_grabbing and self.snap_to and held_idx is not None:
            self.env.snap_piece(held_idx, self.snap_pos, self.snap_rot)
            frame = self.env.render()
            frame = self._overlay_cursor(frame)
        return frame

    def rotate(self, degrees: float) -> np.ndarray:
        self._pending_rotation_deg = float(np.clip(degrees, -180.0, 180.0))
        return self._step_env()

    def finished(self) -> np.ndarray:
        self._finished = True
        action = ActionPoint(
            x=self.cursor.x / self.env.canvas_w,
            y=self.cursor.y / self.env.canvas_h,
            grab=False,
            rotation_delta=0.0,
            finished=True,
        )
        frame = self.env.step({self.cursor_id: action})
        return self._overlay_cursor(frame)

    # ---- Internal ----

    def _step_env(self) -> np.ndarray:
        rot_norm = self._pending_rotation_deg / 180.0
        action = ActionPoint(
            x=self.cursor.x / self.env.canvas_w,
            y=self.cursor.y / self.env.canvas_h,
            grab=self.cursor.grabbing,
            rotation_delta=rot_norm,
            render_priority=1.0,
        )
        self._pending_rotation_deg = 0.0
        frame = self.env.step({self.cursor_id: action})
        return self._overlay_cursor(frame)

    def _overlay_cursor(self, frame: np.ndarray) -> np.ndarray:
        img = Image.fromarray(frame).convert("RGBA")
        draw = ImageDraw.Draw(img)
        x, y = self.cursor.x, self.cursor.y
        r = 10
        outline = (255, 255, 255, 255)
        fill = (*self.cursor_color, 220 if self.cursor.grabbing else 140)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=2)
        # Small crosshair
        draw.line([x - r - 4, y, x + r + 4, y], fill=outline, width=1)
        draw.line([x, y - r - 4, x, y + r + 4], fill=outline, width=1)
        if self.cursor.grabbing:
            draw.text((x + r + 4, y - r - 14), "GRAB", fill=outline)
        return np.array(img.convert("RGB"))

    @property
    def is_finished(self) -> bool:
        return self._finished


def benchmark_llm(
    env: JigsawEnvironment,
    agent: LLMAgent,
    *,
    cursor_id: int = 0,
    max_steps: int = 2_000,
    snap_to: bool = True,
    snap_pos_threshold_px: float = 28.0,
    snap_rot_threshold_deg: float = 20.0,
    pos_tol_px: float = 6.0,
    rot_tol_deg: float = 5.0,
    initial_observation_only: bool = False,
) -> BenchmarkResult:
    """Drive a vanilla LLM agent through the puzzle using the cursor interface."""
    iface = LLMCursorInterface(
        env, cursor_id=cursor_id, snap_to=snap_to,
        snap_pos_threshold_px=snap_pos_threshold_px,
        snap_rot_threshold_deg=snap_rot_threshold_deg,
    )
    env.reset()
    obs = iface._overlay_cursor(env.render())
    t0 = time.perf_counter()

    for step in range(1, max_steps + 1):
        try:
            tool, kwargs = agent(obs, step, TOOL_SCHEMA)
        except Exception:  # noqa: BLE001 — surface as failure
            obs = iface._overlay_cursor(env.render())
            break

        if tool == "move":
            obs = iface.move(float(kwargs.get("dx", 0)), float(kwargs.get("dy", 0)))
        elif tool == "grab":
            obs = iface.grab()
        elif tool == "release":
            obs = iface.release()
        elif tool == "rotate":
            obs = iface.rotate(float(kwargs.get("degrees", 0)))
        elif tool == "finished":
            obs = iface.finished()
            break
        else:
            # Unknown tool: no-op step
            obs = iface.move(0, 0)

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
    )
