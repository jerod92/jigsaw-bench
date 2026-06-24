"""Geometric observation interface for structured (non-visual) deep learning models.

A *geometric* model receives the current puzzle state as structured arrays derived
purely from piece/target geometry — no pixel data — and outputs an action per cursor.
This makes the observation space far smaller than pixel observations and decouples the
model from rendering details.

Observation schema
------------------
For a puzzle with **N** pieces the observation contains a pair of float32 arrays:

``piece_features`` : (N, GEO_PIECE_DIM)
    One row per piece, in ascending piece-index order.

    col  feature
    ---  -------
    0    cx / canvas_w          current centroid x, normalised to [0, 1]
    1    cy / canvas_h          current centroid y
    2    sin(rotation_deg)      rotation encoded as unit-circle coords
    3    cos(rotation_deg)
    4    tx / canvas_w          target centroid x
    5    ty / canvas_h          target centroid y
    6    (tx - cx) / canvas_w   positional delta x (target − current)
    7    (ty - cy) / canvas_h   positional delta y

``cursor_features`` : (K, GEO_CURSOR_DIM)
    One row per *active* cursor (K ≥ 1; a dummy all-zeros row is included when
    the environment has no cursors yet so the tensor shape is always valid).

    col  feature
    ---  -------
    0    last_x / canvas_w      most recent cursor x, normalised
    1    last_y / canvas_h      most recent cursor y
    2    is_holding              1.0 if cursor holds a piece, else 0.0
    3    held_cx / canvas_w     centroid x of held piece (0.0 if not holding)
    4    held_cy / canvas_h     centroid y of held piece (0.0 if not holding)

Supporting arrays
-----------------
``piece_indices`` : (N,) int32  — maps each row of ``piece_features`` to its piece index.
``cursor_ids``    : (K,) int32  — maps each row of ``cursor_features`` to its cursor id.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import JigsawEnvironment


GEO_PIECE_DIM: int = 8
GEO_CURSOR_DIM: int = 5


@dataclass
class GeoObservation:
    """Structured geometric observation extracted from a live :class:`JigsawEnvironment`."""

    piece_features: np.ndarray   # (N, GEO_PIECE_DIM) float32
    cursor_features: np.ndarray  # (K, GEO_CURSOR_DIM) float32
    piece_indices: np.ndarray    # (N,) int32 — row → piece index
    cursor_ids: np.ndarray       # (K,) int32 — row → cursor id
    canvas_w: int
    canvas_h: int


def geo_observation(env: JigsawEnvironment) -> GeoObservation:
    """Extract a :class:`GeoObservation` from the current environment state.

    Safe to call at any time — before, during, or after a rollout.  Internally
    uses the same :py:meth:`~JigsawEnvironment.piece_centroids` / rotations /
    targets accessors as the benchmark scorer so the observations are consistent
    with the reward signal.
    """
    cw, ch = env.canvas_w, env.canvas_h
    centroids = env.piece_centroids()
    targets = env.target_centroids()
    rotations = env.piece_rotations()

    indices = sorted(centroids.keys())
    piece_feats = np.zeros((len(indices), GEO_PIECE_DIM), dtype=np.float32)
    for row, idx in enumerate(indices):
        cx, cy = centroids[idx]
        tx, ty = targets[idx]
        rot = rotations[idx]
        piece_feats[row] = [
            cx / cw,
            cy / ch,
            np.sin(np.radians(rot)),
            np.cos(np.radians(rot)),
            tx / cw,
            ty / ch,
            (tx - cx) / cw,
            (ty - cy) / ch,
        ]

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
        piece_features=piece_feats,
        cursor_features=cursor_feats,
        piece_indices=np.array(indices, dtype=np.int32),
        cursor_ids=np.array(cursor_ids_list if cursor_ids_list else [0], dtype=np.int32),
        canvas_w=cw,
        canvas_h=ch,
    )
