"""Geometric observation interface for structured deep learning models.

A *geometric* model receives two inputs per step:

1. **The rendered canvas** (``frame``) — the same RGB image a human would see.
   The model must infer piece positions, shapes, and targets visually.

2. **Cursor features** (``cursor_features``) — structured state for each
   active cursor (position, grab state, held-piece centroid).  This is the
   *only* geometric data the model gets; it does **not** receive raw piece
   coordinates or target positions directly.

Piece geometry (centroids, rotations, target positions) is kept internal to
the environment and is used by oracles, reward functions, and the benchmark
scorer — but is intentionally hidden from the model so that models must
actually learn to interpret the rendered frame.

``GeoObservation``
------------------
::

    obs.frame            (H, W, 3) uint8   — full-resolution RGB canvas
    obs.cursor_features  (K, 5)   float32  — per-cursor state (see below)
    obs.cursor_ids       (K,)     int32    — maps each row → cursor id
    obs.canvas_w / .h   int               — canvas dimensions in pixels

``cursor_features`` columns (``GEO_CURSOR_DIM = 5``):

    col  feature
    ---  -------
    0    last_x / canvas_w          normalised cursor x, most recent step
    1    last_y / canvas_h          normalised cursor y
    2    is_holding                  1.0 if cursor currently holds a piece
    3    held_cx / canvas_w         centroid x of held piece (0 if none)
    4    held_cy / canvas_h         centroid y of held piece (0 if none)

Internal helpers
----------------
``piece_geo_features(env)`` returns ``(N, GEO_PIECE_DIM)`` float32 with per-piece
geometry for use by oracles and reward functions.  It is **not** part of
``GeoObservation`` and should not be passed to the model.

``GEO_PIECE_DIM = 8`` columns (oracle / reward use only):

    col  feature
    ---  -------
    0–1  cx/W, cy/H          current centroid (normalised)
    2–3  sin(rot), cos(rot)  rotation as unit-circle coords
    4–5  tx/W, ty/H          target (solved) centroid
    6–7  Δx/W, Δy/H         (target − current) positional delta
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import JigsawEnvironment


GEO_CURSOR_DIM: int = 5   # columns in cursor_features
GEO_PIECE_DIM: int = 8    # columns in piece_geo_features() — oracle/reward only


@dataclass
class GeoObservation:
    """Observation for a geometric DL model: rendered frame + cursor state.

    Piece positions and target locations are **not** included here; models
    must infer them visually from ``frame``.
    """

    frame: np.ndarray            # (H, W, 3) uint8 rendered canvas
    cursor_features: np.ndarray  # (K, GEO_CURSOR_DIM) float32
    cursor_ids: np.ndarray       # (K,) int32 — row → cursor id
    canvas_w: int
    canvas_h: int


def geo_observation(env: JigsawEnvironment, frame: np.ndarray | None = None) -> GeoObservation:
    """Extract a :class:`GeoObservation` from *env*.

    Parameters
    ----------
    env:
        Live environment to read cursor state from.
    frame:
        Pre-rendered RGB frame ``(H, W, 3) uint8``.  Pass the frame you
        already have from the last :py:meth:`~JigsawEnvironment.step` call to
        avoid rendering twice.  If *None*, ``env.render()`` is called.
    """
    if frame is None:
        frame = env.render()

    cw, ch = env.canvas_w, env.canvas_h
    centroids = env.piece_centroids()

    cursor_ids_list = sorted(env.cursors.keys()) if env.cursors else []
    n_cursors = max(1, len(cursor_ids_list))
    cursor_feats = np.zeros((n_cursors, GEO_CURSOR_DIM), dtype=np.float32)

    for row, cid in enumerate(cursor_ids_list):
        cs = env.cursors[cid]
        held_cx, held_cy, is_holding = 0.0, 0.0, 0.0
        if cs.held_piece is not None and cs.held_piece in centroids:
            hcx, hcy = centroids[cs.held_piece]
            held_cx, held_cy = hcx / cw, hcy / ch
            is_holding = 1.0
        cursor_feats[row] = [
            cs.last_x / cw,
            cs.last_y / ch,
            is_holding,
            held_cx,
            held_cy,
        ]

    return GeoObservation(
        frame=frame,
        cursor_features=cursor_feats,
        cursor_ids=np.array(cursor_ids_list if cursor_ids_list else [0], dtype=np.int32),
        canvas_w=cw,
        canvas_h=ch,
    )


def piece_geo_features(env: JigsawEnvironment) -> np.ndarray:
    """Return ``(N, GEO_PIECE_DIM)`` float32 of per-piece geometry.

    Intended for oracles and reward functions, **not** as model input.
    Rows are in ascending piece-index order; use ``sorted(env.pieces.keys())``
    to map row indices back to piece indices.
    """
    cw, ch = env.canvas_w, env.canvas_h
    centroids = env.piece_centroids()
    targets = env.target_centroids()
    rotations = env.piece_rotations()

    indices = sorted(centroids.keys())
    feats = np.zeros((len(indices), GEO_PIECE_DIM), dtype=np.float32)
    for row, idx in enumerate(indices):
        cx, cy = centroids[idx]
        tx, ty = targets[idx]
        rot = rotations[idx]
        feats[row] = [
            cx / cw,
            cy / ch,
            np.sin(np.radians(rot)),
            np.cos(np.radians(rot)),
            tx / cw,
            ty / ch,
            (tx - cx) / cw,
            (ty - cy) / ch,
        ]
    return feats
