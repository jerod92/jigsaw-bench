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

from .puzzle import Piece, Puzzle


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
    """Worst-case rotated-bbox diagonal across all pieces and allowed rotations."""
    worst = 0.0
    for p in pieces:
        h, w = p.sprite_rgba.shape[:2]
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
    canvas_scale: float = 2.2,
    margin: int = 24,
    rotation_deg_choices: tuple[float, ...] | None = None,
    seed: int | None = None,
    cell_padding: float = 1.5,
) -> ShuffleLayout:
    """Scatter pieces around a centered silhouette using deterministic concentric rings.

    Cell size = worst-case rotated-bbox extent across all pieces × ``cell_padding``.
    The board is laid out in the canvas center; pieces fill rings starting just outside
    the silhouette and growing outward. The canvas is automatically expanded if the
    requested ``canvas_scale`` doesn't fit all pieces.
    """
    rng = np.random.default_rng(seed)
    rotation_choices = rotation_deg_choices or tuple(np.arange(0, 360, 15.0))
    cell = int(math.ceil(_max_rotated_extent(puzzle.pieces, rotation_choices) * cell_padding))

    cw = max(int(puzzle.width * canvas_scale), puzzle.width + 2 * (margin + cell * 3))
    ch = max(int(puzzle.height * canvas_scale), puzzle.height + 2 * (margin + cell * 3))
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
        angle = float(rng.choice(rotation_choices))
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
