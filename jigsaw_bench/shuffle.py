"""Shuffle pieces around a centered silhouette in a larger working canvas.

Layout requirements:
- The puzzle silhouette is centered (silhouette = grayed image of where pieces belong).
- Pieces are placed only in the surrounding margin, never on the silhouette.
- No two pieces overlap in their scattered location (rotated bbox check).
- Each piece carries metadata: target centroid, shuffle centroid, rotation degrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .puzzle import Puzzle


@dataclass
class ShuffleLayout:
    canvas_width: int
    canvas_height: int
    board_origin: tuple[int, int]                          # (x, y) of the silhouette's top-left
    board_size: tuple[int, int]                            # (w, h) of the silhouette
    placements: list["PiecePlacement"] = field(default_factory=list)


@dataclass
class PiecePlacement:
    index: int
    target_centroid: tuple[float, float]                   # within the silhouette (board) frame, offset to canvas
    shuffle_centroid: tuple[float, float]                  # canvas coords
    rotation_deg: float
    aabb_canvas: tuple[float, float, float, float]         # rotated piece's axis-aligned bbox on the canvas


def _rotated_aabb(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    """Side lengths of the AABB of a rotated wxh rectangle."""
    a = math.radians(angle_deg)
    cw = abs(w * math.cos(a)) + abs(h * math.sin(a))
    ch = abs(w * math.sin(a)) + abs(h * math.cos(a))
    return cw, ch


def _aabb_at_centroid(cx: float, cy: float, sprite_w: int, sprite_h: int, angle_deg: float, pad: float = 4.0):
    cw, ch = _rotated_aabb(sprite_w, sprite_h, angle_deg)
    cw += pad
    ch += pad
    return (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)


def _rects_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _max_rotated_extent(pieces, rotation_choices) -> int:
    """Worst-case rotated-bbox extent across all pieces and rotation choices.

    For continuous rotations, worst case is at 45 deg: ``(w + h) / sqrt(2)``. For a
    discrete set, sample each angle. Either way, take the max side across all pieces.
    """
    worst = 0.0
    for p in pieces:
        h, w = p.sprite_rgba.shape[:2]
        if rotation_choices is None:
            # Continuous: 45 deg gives the largest AABB side for any rectangle.
            rw, rh = _rotated_aabb(w, h, 45.0)
            worst = max(worst, max(rw, rh))
        else:
            for a in rotation_choices:
                rw, rh = _rotated_aabb(w, h, a)
                worst = max(worst, max(rw, rh))
    return int(math.ceil(worst))


def _ring_slots(silhouette_aabb, canvas_w, canvas_h, cell, ring_idx):
    """Yield evenly-spaced cell-center positions along ring_idx's perimeter.

    Positions are spaced by ``cell`` along perimeter arc-length, so corners don't pile up.
    """
    sx0, sy0, sx1, sy1 = silhouette_aabb
    pad = ring_idx * cell - cell / 2  # ring 1 sits one cell out from silhouette
    x0 = sx0 - pad
    x1 = sx1 + pad
    y0 = sy0 - pad
    y1 = sy1 + pad
    if x0 < cell / 2 or y0 < cell / 2 or x1 > canvas_w - cell / 2 or y1 > canvas_h - cell / 2:
        return

    side_w = x1 - x0
    side_h = y1 - y0
    perim = 2 * (side_w + side_h)
    n = max(4, int(perim // cell))
    step = perim / n

    for k in range(n):
        s = (k + 0.5) * step  # offset by half-step so first slot isn't on a corner
        if s < side_w:
            yield (x0 + s, y0)
        elif s < side_w + side_h:
            yield (x1, y0 + (s - side_w))
        elif s < 2 * side_w + side_h:
            yield (x1 - (s - side_w - side_h), y1)
        else:
            yield (x0, y1 - (s - 2 * side_w - side_h))


def shuffle_pieces(
    puzzle: Puzzle,
    canvas_scale: float = 3.0,
    margin: int = 24,
    rotation_deg_choices: tuple[float, ...] | None = None,
    seed: int | None = None,
    cell_padding: float = 1.5,
) -> ShuffleLayout:
    """Scatter pieces around a centered silhouette using deterministic concentric rings.

    By default each piece's rotation is drawn from a **continuous uniform distribution
    over [0, 360)**. Pass ``rotation_deg_choices`` (e.g. ``(0, 90, 180, 270)``) to use
    a discrete set instead — useful for snap-friendly easy modes.

    ``canvas_scale`` is the **target canvas-to-puzzle ratio** (default 3.0 — the
    rendered board is ~1/3 of canvas dims). Aspect ratio of the puzzle is preserved.
    The canvas auto-grows beyond this if pieces don't fit at the requested scale,
    but never shrinks below it.

    Cell size = worst-case rotated-bbox extent across all pieces × ``cell_padding``.
    """
    rng = np.random.default_rng(seed)
    cell = int(math.ceil(_max_rotated_extent(puzzle.pieces, rotation_deg_choices) * cell_padding))

    cw = max(int(puzzle.width * canvas_scale), puzzle.width + 2 * (margin + cell))
    ch = max(int(puzzle.height * canvas_scale), puzzle.height + 2 * (margin + cell))
    bx = (cw - puzzle.width) // 2
    by = (ch - puzzle.height) // 2
    silhouette_aabb = (bx - margin, by - margin, bx + puzzle.width + margin, by + puzzle.height + margin)

    # Generate ring slots until we have enough; expand canvas if needed.
    pieces_sorted = sorted(puzzle.pieces, key=lambda p: -(p.sprite_rgba.shape[0] * p.sprite_rgba.shape[1]))
    n = len(pieces_sorted)
    slots: list[tuple[float, float]] = []
    ring = 1
    while len(slots) < n:
        ring_slots = list(_ring_slots(silhouette_aabb, cw, ch, cell, ring))
        if not ring_slots and ring > 1:
            # Out of room — grow canvas.
            cw += cell * 2
            ch += cell * 2
            bx = (cw - puzzle.width) // 2
            by = (ch - puzzle.height) // 2
            silhouette_aabb = (bx - margin, by - margin, bx + puzzle.width + margin, by + puzzle.height + margin)
            slots.clear()
            ring = 1
            continue
        slots.extend(ring_slots)
        ring += 1
        if ring > 64:
            raise RuntimeError("Layout failed: too many rings — pieces are larger than canvas can hold.")

    # Shuffle slot assignment so pieces aren't laid out in order.
    perm = rng.permutation(len(slots))[:n]
    placements: list[PiecePlacement] = []
    for piece, slot_idx in zip(pieces_sorted, perm):
        cx, cy = slots[slot_idx]
        # Mild jitter that stays inside the cell's safety margin (no overlap possible).
        slack = max(0.0, cell * (1.0 - 1.0 / cell_padding) * 0.45)
        cx += rng.uniform(-slack, slack)
        cy += rng.uniform(-slack, slack)
        if rotation_deg_choices is None:
            angle = float(rng.uniform(0.0, 360.0))
        else:
            angle = float(rng.choice(rotation_deg_choices))
        sh, sw = piece.sprite_rgba.shape[:2]
        aabb = _aabb_at_centroid(cx, cy, sw, sh, angle)
        tcx, tcy = piece.target_centroid
        placements.append(
            PiecePlacement(
                index=piece.index,
                target_centroid=(float(tcx + bx), float(tcy + by)),
                shuffle_centroid=(float(cx), float(cy)),
                rotation_deg=angle,
                aabb_canvas=aabb,
            )
        )

    placements.sort(key=lambda p: p.index)
    return ShuffleLayout(
        canvas_width=cw,
        canvas_height=ch,
        board_origin=(bx, by),
        board_size=(puzzle.width, puzzle.height),
        placements=placements,
    )
