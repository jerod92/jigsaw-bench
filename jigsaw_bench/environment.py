"""Interactive jigsaw environment.

Pieces have continuous position + rotation. Any number of "cursors" can grab pieces at
absolute canvas coordinates. Each cursor describes: where it is, whether it is grabbing,
how much to rotate the held piece, a render priority that controls layering between
simultaneously held pieces, and whether it's finished (no longer participating).

The environment is headless-friendly: render() returns an RGB ndarray. There is no
window/event loop — drive it from Python.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from .puzzle import Puzzle
from .shuffle import ShuffleLayout


@dataclass
class ActionPoint:
    """One cursor's action for a single timestep.

    Coordinates are normalized to (0, 1) with respect to canvas (cursor.x * canvas_w, etc.).
    rotation_delta is in (-1, +1) corresponding to (-180°, +180°) per step (capped per call).
    """
    x: float
    y: float
    grab: bool
    rotation_delta: float = 0.0
    render_priority: float = 0.0
    finished: bool = False


@dataclass
class _PieceState:
    index: int
    sprite_rgba: np.ndarray                      # (h, w, 4) uint8 — original (unrotated) sprite
    sprite_centroid_local: tuple[float, float]   # the local-pixel coords of the piece centroid (anchor for rotation)
    centroid: tuple[float, float]                # canvas-space centroid
    rotation_deg: float                          # canvas-space rotation
    target_centroid: tuple[float, float]
    z: float = 0.0                               # base z order; grabbed pieces are bumped


@dataclass
class _CursorState:
    cursor_id: int
    held_piece: int | None = None
    grab_offset_local: tuple[float, float] | None = None   # (in piece-local rotated frame) where the cursor is on the piece, relative to piece centroid
    last_x: float = 0.0
    last_y: float = 0.0
    finished: bool = False
    render_priority: float = 0.0


class JigsawEnvironment:
    """Stateful, headless jigsaw environment.

    Multiple cursors can act in parallel each step. Each cursor independently grabs/holds
    pieces. The environment renders the canvas as an RGB image.
    """

    def __init__(self, puzzle: Puzzle, layout: ShuffleLayout, *, max_step_rotation_deg: float = 180.0,
                 background: tuple[int, int, int] = (32, 32, 32),
                 initial_cursor_positions: dict[int, tuple[float, float]] | None = None):
        self.puzzle = puzzle
        self.layout = layout
        self.canvas_w = layout.canvas_width
        self.canvas_h = layout.canvas_height
        self.background = background
        self.max_step_rotation_deg = max_step_rotation_deg
        # Optional spread-out starting positions (canvas px) so cursors are
        # distinguishable from the very first step — this makes "move toward
        # your nearest piece" a well-posed function of (frame, cursor state).
        self.initial_cursor_positions = dict(initial_cursor_positions or {})
        self._step = 0
        self._next_z = 0.0

        bx, by = layout.board_origin
        self._board_origin = (bx, by)

        # Build per-piece state from layout.
        idx_to_piece = {p.index: p for p in puzzle.pieces}
        self.pieces: dict[int, _PieceState] = {}
        for placement in sorted(layout.placements, key=lambda p: p.index):
            piece = idx_to_piece[placement.index]
            sprite = piece.sprite_rgba
            x0, y0, x1, y1 = piece.bbox
            cx_local = piece.target_centroid[0] - x0
            cy_local = piece.target_centroid[1] - y0
            self.pieces[placement.index] = _PieceState(
                index=placement.index,
                sprite_rgba=sprite,
                sprite_centroid_local=(float(cx_local), float(cy_local)),
                centroid=placement.shuffle_centroid,
                rotation_deg=placement.rotation_deg,
                target_centroid=placement.target_centroid,
                z=self._next_z,
            )
            self._next_z += 1.0

        self.cursors: dict[int, _CursorState] = {}

    # ---------- Public API ----------

    def reset(self) -> np.ndarray:
        for placement in self.layout.placements:
            ps = self.pieces[placement.index]
            ps.centroid = placement.shuffle_centroid
            ps.rotation_deg = placement.rotation_deg
        self.cursors.clear()
        # Pre-create cursors at their spread-out starting positions, if given.
        for cid, (x, y) in self.initial_cursor_positions.items():
            cs = _CursorState(cursor_id=cid)
            cs.last_x, cs.last_y = float(x), float(y)
            self.cursors[cid] = cs
        self._step = 0
        return self.render()

    def step(self, actions: dict[int, ActionPoint] | Sequence[ActionPoint]) -> np.ndarray:
        """Apply one action per cursor; return new rendered RGB image."""
        if isinstance(actions, dict):
            items = list(actions.items())
        else:
            items = list(enumerate(actions))

        for cid, action in items:
            self._apply_cursor(cid, action)
        self._step += 1
        return self.render()

    def render(self) -> np.ndarray:
        """Composite the canvas. Returns an HxWx3 uint8 array."""
        canvas = Image.new("RGBA", (self.canvas_w, self.canvas_h),
                           (*self.background, 255))
        # Draw silhouette (board) area as a slightly lighter rectangle for visibility.
        bx, by = self._board_origin
        bw, bh = self.layout.board_size
        silhouette = Image.new("RGBA", (bw, bh), (64, 64, 64, 255))
        canvas.paste(silhouette, (bx, by))

        # Order pieces: held pieces last (top); within held set, order by render_priority.
        held_priority = {c.held_piece: c.render_priority for c in self.cursors.values()
                         if c.held_piece is not None and not c.finished}
        free = [p for p in self.pieces.values() if p.index not in held_priority]
        held = [p for p in self.pieces.values() if p.index in held_priority]
        free.sort(key=lambda p: p.z)
        held.sort(key=lambda p: held_priority[p.index])

        for ps in free + held:
            self._draw_piece(canvas, ps)

        return np.array(canvas.convert("RGB"))

    # ---------- Internal ----------

    def _apply_cursor(self, cid: int, action: ActionPoint) -> None:
        cs = self.cursors.get(cid)
        if cs is None:
            cs = _CursorState(cursor_id=cid)
            self.cursors[cid] = cs
        if cs.finished:
            return

        # Convert normalized to canvas pixels.
        cx = float(np.clip(action.x, 0.0, 1.0)) * self.canvas_w
        cy = float(np.clip(action.y, 0.0, 1.0)) * self.canvas_h
        cs.last_x, cs.last_y = cx, cy
        cs.render_priority = float(np.clip(action.render_priority, 0.0, 1.0))

        if action.finished:
            self._release(cs)
            cs.finished = True
            return

        if action.grab:
            if cs.held_piece is None:
                self._try_grab(cs, cx, cy)
            else:
                self._update_held(cs, cx, cy, action.rotation_delta)
        else:
            if cs.held_piece is not None:
                self._release(cs)

    def _try_grab(self, cs: _CursorState, cx: float, cy: float) -> None:
        """Find topmost *unheld* piece whose mask covers (cx, cy) and grab it.

        Pieces already held by another cursor are skipped — a held piece is
        claimed and cannot be stolen, so a carried piece passing over another
        piece's grab point won't be picked up by a second cursor.
        """
        held_by_others = {
            c.held_piece for c in self.cursors.values()
            if c is not cs and c.held_piece is not None
        }
        # Iterate from highest z to lowest.
        ordered = sorted(self.pieces.values(), key=lambda p: -p.z)
        for ps in ordered:
            if ps.index in held_by_others:
                continue
            if self._point_hits_piece(ps, cx, cy):
                cs.held_piece = ps.index
                # Grab offset = where this cursor sits, expressed in the piece's local frame
                # (i.e. before rotation). We'll re-apply rotation each frame.
                dx = cx - ps.centroid[0]
                dy = cy - ps.centroid[1]
                a = math.radians(-ps.rotation_deg)
                lx = dx * math.cos(a) - dy * math.sin(a)
                ly = dx * math.sin(a) + dy * math.cos(a)
                cs.grab_offset_local = (lx, ly)
                # Bump z: held pieces always render on top, layering by priority handled at render time.
                self._next_z += 1.0
                ps.z = self._next_z
                return

    def _release(self, cs: _CursorState) -> None:
        cs.held_piece = None
        cs.grab_offset_local = None

    def _update_held(self, cs: _CursorState, cx: float, cy: float, rotation_delta_norm: float) -> None:
        ps = self.pieces[cs.held_piece]
        # Rotate about the grab point (kept fixed at the cursor location).
        delta_deg = float(np.clip(rotation_delta_norm, -1.0, 1.0)) * self.max_step_rotation_deg
        new_rot = ps.rotation_deg + delta_deg

        # Compute world-space grab point under new rotation, then translate so it lands at (cx, cy).
        a = math.radians(new_rot)
        lx, ly = cs.grab_offset_local
        gx_rel = lx * math.cos(a) - ly * math.sin(a)
        gy_rel = lx * math.sin(a) + ly * math.cos(a)
        ps.centroid = (cx - gx_rel, cy - gy_rel)
        ps.rotation_deg = new_rot

    def _point_hits_piece(self, ps: _PieceState, cx: float, cy: float) -> bool:
        """Test (cx, cy) against the piece's rotated alpha mask."""
        sprite = ps.sprite_rgba
        h, w = sprite.shape[:2]
        local_cx, local_cy = ps.sprite_centroid_local
        # Inverse-transform (cx, cy) into piece-local pixel coords.
        dx = cx - ps.centroid[0]
        dy = cy - ps.centroid[1]
        a = math.radians(-ps.rotation_deg)
        lx = dx * math.cos(a) - dy * math.sin(a) + local_cx
        ly = dx * math.sin(a) + dy * math.cos(a) + local_cy
        ix, iy = int(round(lx)), int(round(ly))
        if not (0 <= ix < w and 0 <= iy < h):
            return False
        return bool(sprite[iy, ix, 3] > 0)

    def _draw_piece(self, canvas: Image.Image, ps: _PieceState) -> None:
        sprite = Image.fromarray(ps.sprite_rgba, mode="RGBA")
        sw, sh = sprite.size
        # Rotate around the SPRITE CENTER (PIL clips when expand=True is combined with a
        # custom center — it sizes the output as if rotating around center anyway).
        rotated = sprite.rotate(ps.rotation_deg, resample=Image.BILINEAR, expand=True)
        rw, rh = rotated.size

        # The piece centroid was at (local_cx, local_cy) in the original sprite. After a
        # rotate-around-center, it lands at the new image's center plus the rotated offset
        # from sprite-center to centroid.
        local_cx, local_cy = ps.sprite_centroid_local
        dx = local_cx - sw / 2
        dy = local_cy - sh / 2
        # PIL rotates CCW by `rotation_deg` in image-y-down coords, which is screen-CW.
        a = math.radians(-ps.rotation_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        new_centroid_x = rw / 2 + dx * cos_a - dy * sin_a
        new_centroid_y = rh / 2 + dx * sin_a + dy * cos_a

        paste_x = int(round(ps.centroid[0] - new_centroid_x))
        paste_y = int(round(ps.centroid[1] - new_centroid_y))
        canvas.alpha_composite(rotated, dest=(paste_x, paste_y))

    # ---------- Convenience for benchmarking ----------

    def piece_centroids(self) -> dict[int, tuple[float, float]]:
        return {idx: ps.centroid for idx, ps in self.pieces.items()}

    def piece_rotations(self) -> dict[int, float]:
        return {idx: ps.rotation_deg for idx, ps in self.pieces.items()}

    def target_centroids(self) -> dict[int, tuple[float, float]]:
        return {idx: ps.target_centroid for idx, ps in self.pieces.items()}

    def snap_piece(self, piece_index: int, threshold_px: float = 20.0, rotation_threshold_deg: float = 12.0) -> bool:
        """If the piece is within thresholds of its solved pose, snap exactly into place."""
        ps = self.pieces[piece_index]
        tx, ty = ps.target_centroid
        dist = math.hypot(ps.centroid[0] - tx, ps.centroid[1] - ty)
        rot_err = ((ps.rotation_deg + 180) % 360) - 180
        if dist <= threshold_px and abs(rot_err) <= rotation_threshold_deg:
            ps.centroid = (tx, ty)
            ps.rotation_deg = 0.0
            return True
        return False

    def is_solved(self, pos_tol_px: float = 4.0, rot_tol_deg: float = 3.0) -> bool:
        for ps in self.pieces.values():
            tx, ty = ps.target_centroid
            if math.hypot(ps.centroid[0] - tx, ps.centroid[1] - ty) > pos_tol_px:
                return False
            rot_err = ((ps.rotation_deg + 180) % 360) - 180
            if abs(rot_err) > rot_tol_deg:
                return False
        return True
