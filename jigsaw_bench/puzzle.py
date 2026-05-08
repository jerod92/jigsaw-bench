"""Build puzzle pieces (image + alpha + polygon + metadata) from a cut-set and image."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .cuts import PuzzleCuts, generate_cuts
from .geometry import piece_polygon, rasterize_polygon


@dataclass
class Piece:
    index: int                          # global piece index
    grid_row: int
    grid_col: int
    polygon: Polygon                    # in canvas coordinates
    bbox: tuple[int, int, int, int]     # (x0, y0, x1, y1) on the source canvas
    sprite_rgba: np.ndarray             # (h, w, 4) uint8 — image+alpha cropped to bbox
    target_centroid: tuple[float, float]  # canvas-space centroid at the solved location


@dataclass
class Puzzle:
    width: int
    height: int
    n_cols: int
    n_rows: int
    image: np.ndarray                   # (H, W, 3) uint8 source canvas (already resized)
    cuts: PuzzleCuts
    pieces: list[Piece] = field(default_factory=list)


def _segment_top(cuts: PuzzleCuts, j: int, i: int) -> np.ndarray:
    """Top edge of piece (j, i): goes left→right between anchors[j,i] and anchors[j,i+1]."""
    A = cuts.anchors
    if j == 0:
        return np.array([A[0, i], A[0, i + 1]])
    return _slice_horizontal(cuts.h_cuts[j - 1], A[j, i], A[j, i + 1])


def _segment_bottom(cuts: PuzzleCuts, j: int, i: int) -> np.ndarray:
    A = cuts.anchors
    if j == cuts.n_rows - 1:
        return np.array([A[-1, i], A[-1, i + 1]])
    return _slice_horizontal(cuts.h_cuts[j], A[j + 1, i], A[j + 1, i + 1])


def _segment_left(cuts: PuzzleCuts, j: int, i: int) -> np.ndarray:
    A = cuts.anchors
    if i == 0:
        return np.array([A[j, 0], A[j + 1, 0]])
    return _slice_vertical(cuts.v_cuts[i - 1], A[j, i], A[j + 1, i])


def _segment_right(cuts: PuzzleCuts, j: int, i: int) -> np.ndarray:
    A = cuts.anchors
    if i == cuts.n_cols - 1:
        return np.array([A[j, -1], A[j + 1, -1]])
    return _slice_vertical(cuts.v_cuts[i], A[j, i + 1], A[j + 1, i + 1])


def _closest_index(polyline: np.ndarray, point: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(polyline - point, axis=1)))


def _slice_horizontal(line: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    s = _closest_index(line, p_start)
    e = _closest_index(line, p_end)
    if s > e:
        s, e = e, s
        seg = line[s:e + 1][::-1].copy()
    else:
        seg = line[s:e + 1].copy()
    seg[0] = p_start
    seg[-1] = p_end
    return seg


def _slice_vertical(line: np.ndarray, p_start: np.ndarray, p_end: np.ndarray) -> np.ndarray:
    s = _closest_index(line, p_start)
    e = _closest_index(line, p_end)
    if s > e:
        s, e = e, s
        seg = line[s:e + 1][::-1].copy()
    else:
        seg = line[s:e + 1].copy()
    seg[0] = p_start
    seg[-1] = p_end
    return seg


def _piece_sprite(image_rgb: np.ndarray, mask_full: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.where(mask_full)
    if len(xs) == 0:
        return np.zeros((1, 1, 4), dtype=np.uint8), (0, 0, 1, 1)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = image_rgb[y0:y1, x0:x1]
    crop_mask = mask_full[y0:y1, x0:x1]
    sprite = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    sprite[..., :3] = crop
    sprite[..., 3] = crop_mask.astype(np.uint8) * 255
    return sprite, (x0, y0, x1, y1)


def _load_and_fit_image(image: Image.Image | np.ndarray | str, width: int, height: int) -> np.ndarray:
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
    else:
        img = image.convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    return np.array(img)


def generate_puzzle(
    image: Image.Image | np.ndarray | str,
    width: int = 1200,
    height: int = 900,
    n_cols: int = 24,
    n_rows: int = 16,
    seed: int | None = None,
    cuts: PuzzleCuts | None = None,
    profile: Sequence[tuple[float, float, float]] | None = None,
) -> Puzzle:
    """Generate a complete puzzle (cuts + per-piece sprites + metadata) from a single image."""
    image_rgb = _load_and_fit_image(image, width, height)
    if cuts is None:
        cuts = generate_cuts(width, height, n_cols, n_rows, profile=profile, seed=seed)
    else:
        assert cuts.width == width and cuts.height == height

    pieces: list[Piece] = []
    idx = 0
    for j in range(n_rows):
        for i in range(n_cols):
            top = _segment_top(cuts, j, i)
            right = _segment_right(cuts, j, i)
            bottom = _segment_bottom(cuts, j, i)
            left = _segment_left(cuts, j, i)
            poly = piece_polygon(top, right, bottom, left)
            mask = rasterize_polygon(poly, width, height)
            sprite, bbox = _piece_sprite(image_rgb, mask)
            cx, cy = poly.centroid.x, poly.centroid.y
            pieces.append(
                Piece(
                    index=idx,
                    grid_row=j,
                    grid_col=i,
                    polygon=poly,
                    bbox=bbox,
                    sprite_rgba=sprite,
                    target_centroid=(float(cx), float(cy)),
                )
            )
            idx += 1
    return Puzzle(
        width=width,
        height=height,
        n_cols=n_cols,
        n_rows=n_rows,
        image=image_rgb,
        cuts=cuts,
        pieces=pieces,
    )
