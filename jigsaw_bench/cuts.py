"""Jigsaw cut generation with intersection validation.

Cuts are constructed as smoothed polylines between grid anchors. Two distinct cuts
must only ever meet at the grid anchors they share — anywhere else is a defect that
would produce non-simply-connected pieces. We detect and regenerate offending cuts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline

from .geometry import polylines_intersect


# (t along edge in [0,1], perpendicular displacement as fraction of edge length, jitter sigma)
DEFAULT_PROFILE: list[tuple[float, float, float]] = [
    (0.00, 0.00, 0.01),
    (0.38, 0.00, 0.01),
    (0.41, 0.10, 0.01),
    (0.35, 0.20, 0.01),
    (0.50, 0.32, 0.01),
]


@dataclass
class PuzzleCuts:
    width: int
    height: int
    n_cols: int
    n_rows: int
    anchors: np.ndarray  # (n_rows+1, n_cols+1, 2)
    h_cuts: list[np.ndarray]  # length n_rows-1, each polyline spanning a full row
    v_cuts: list[np.ndarray]  # length n_cols-1, each polyline spanning a full column
    profile: list[tuple[float, float, float]] = field(default_factory=lambda: list(DEFAULT_PROFILE))

    def all_polylines(self) -> list[tuple[str, int, np.ndarray]]:
        out: list[tuple[str, int, np.ndarray]] = []
        for j, line in enumerate(self.h_cuts, start=1):
            out.append(("h", j, line))
        for i, line in enumerate(self.v_cuts, start=1):
            out.append(("v", i, line))
        return out


def generate_grid_anchors(
    width: int,
    height: int,
    n_cols: int,
    n_rows: int,
    sigma_x: float = 0.0,
    sigma_y: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    anchors = np.zeros((n_rows + 1, n_cols + 1, 2))
    for j in range(n_rows + 1):
        for i in range(n_cols + 1):
            x = i * width / n_cols
            y = j * height / n_rows
            anchors[j, i] = [
                x + rng.normal(0, sigma_x),
                y + rng.normal(0, sigma_y),
            ]
    anchors[0, :, 1] = 0
    anchors[-1, :, 1] = height
    anchors[:, 0, 0] = 0
    anchors[:, -1, 0] = width
    return anchors


def _normalize_and_mirror_profile(profile):
    half = list(profile)
    second_half = []
    for t, perp, sigma in reversed(profile[:-1]):
        second_half.append((1.0 - t, perp, sigma))
    return half + second_half


def _build_edge_points(p1, p2, base_profile, flip, rng):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return np.array([p1, p2])
    unit_perp = np.array([-dy, dx]) / length
    if flip:
        unit_perp = -unit_perp

    full_profile = _normalize_and_mirror_profile(base_profile)
    points = []
    for t, perp_off, sigma in full_profile:
        jitter = rng.normal(0, sigma)
        pos = p1 + t * (p2 - p1)
        perp_disp = (perp_off + jitter) * length * unit_perp
        points.append(pos + perp_disp)
    return np.array(points)


def _smooth_spline(points, num_samples=120):
    if len(points) < 3:
        return np.asarray(points, dtype=float)
    diffs = np.diff(points, axis=0)
    t = np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))
    t = np.insert(t, 0, 0.0)
    if t[-1] <= 0:
        return np.asarray(points, dtype=float)
    t = t / t[-1]
    cs_x = CubicSpline(t, points[:, 0], bc_type="clamped")
    cs_y = CubicSpline(t, points[:, 1], bc_type="clamped")
    t_fine = np.linspace(0, 1, num_samples)
    return np.stack([cs_x(t_fine), cs_y(t_fine)], axis=1)


def _build_horizontal_polyline(anchors, j, profile, rng, samples_per_edge=60):
    n_cols = anchors.shape[1] - 1
    pieces = []
    for i in range(n_cols):
        p1 = anchors[j, i]
        p2 = anchors[j, i + 1]
        flip = rng.random() < 0.5
        edge_pts = _build_edge_points(p1, p2, profile, flip, rng)
        smooth = _smooth_spline(edge_pts, num_samples=samples_per_edge)
        # Snap endpoints to anchors so adjacent cuts kiss exactly.
        smooth[0] = p1
        smooth[-1] = p2
        if i == 0:
            pieces.append(smooth)
        else:
            pieces.append(smooth[1:])
    return np.concatenate(pieces, axis=0)


def _build_vertical_polyline(anchors, i, profile, rng, samples_per_edge=60):
    n_rows = anchors.shape[0] - 1
    pieces = []
    for j in range(n_rows):
        p1 = anchors[j, i]
        p2 = anchors[j + 1, i]
        flip = rng.random() < 0.5
        edge_pts = _build_edge_points(p1, p2, profile, flip, rng)
        smooth = _smooth_spline(edge_pts, num_samples=samples_per_edge)
        smooth[0] = p1
        smooth[-1] = p2
        if j == 0:
            pieces.append(smooth)
        else:
            pieces.append(smooth[1:])
    return np.concatenate(pieces, axis=0)


def _shared_anchor_set(kind_a, idx_a, kind_b, idx_b, anchors):
    """Return the list of grid anchor points shared by two cuts, where touching is allowed."""
    n_rows, n_cols = anchors.shape[0] - 1, anchors.shape[1] - 1
    if kind_a == "h" and kind_b == "h":
        return []  # parallel cuts share no anchors
    if kind_a == "v" and kind_b == "v":
        return []
    # h × v: they share the single anchor at (h_row, v_col)
    h_row = idx_a if kind_a == "h" else idx_b
    v_col = idx_b if kind_a == "h" else idx_a
    return [anchors[h_row, v_col]]


def _cuts_intersect_other_cuts(
    polylines: list[tuple[str, int, np.ndarray]],
    target_index: int,
    anchors: np.ndarray,
) -> bool:
    """Check whether cut at target_index crosses any other cut outside shared anchors."""
    kind_a, idx_a, line_a = polylines[target_index]
    for k, (kind_b, idx_b, line_b) in enumerate(polylines):
        if k == target_index:
            continue
        shared = _shared_anchor_set(kind_a, idx_a, kind_b, idx_b, anchors)
        if polylines_intersect(line_a, line_b, shared_endpoints=shared):
            return True
    return False


def generate_cuts(
    width: int = 1200,
    height: int = 900,
    n_cols: int = 24,
    n_rows: int = 16,
    profile: Sequence[tuple[float, float, float]] | None = None,
    anchor_sigma_x: float = 3.0,
    anchor_sigma_y: float = 3.0,
    seed: int | None = None,
    max_retries_per_cut: int = 50,
    samples_per_edge: int = 60,
) -> PuzzleCuts:
    """Generate jigsaw cuts; regenerate any individual cut that intersects another.

    If a single cut still intersects after `max_retries_per_cut` rebuilds, raises RuntimeError.
    """
    rng = np.random.default_rng(seed)
    profile = list(profile or DEFAULT_PROFILE)

    anchors = generate_grid_anchors(width, height, n_cols, n_rows, anchor_sigma_x, anchor_sigma_y, rng)

    h_cuts = [
        _build_horizontal_polyline(anchors, j, profile, rng, samples_per_edge)
        for j in range(1, n_rows)
    ]
    v_cuts = [
        _build_vertical_polyline(anchors, i, profile, rng, samples_per_edge)
        for i in range(1, n_cols)
    ]

    # Validate: for each cut, if it intersects any other cut outside shared anchors,
    # rebuild that single cut up to max_retries_per_cut times.
    polylines: list[tuple[str, int, np.ndarray]] = []
    for j, line in enumerate(h_cuts, start=1):
        polylines.append(("h", j, line))
    for i, line in enumerate(v_cuts, start=1):
        polylines.append(("v", i, line))

    for k, (kind, idx, _line) in enumerate(polylines):
        attempts = 0
        while _cuts_intersect_other_cuts(polylines, k, anchors):
            attempts += 1
            if attempts > max_retries_per_cut:
                raise RuntimeError(
                    f"Could not generate a non-intersecting {kind}-cut at index {idx} "
                    f"after {max_retries_per_cut} retries. Try gentler profile."
                )
            if kind == "h":
                new = _build_horizontal_polyline(anchors, idx, profile, rng, samples_per_edge)
            else:
                new = _build_vertical_polyline(anchors, idx, profile, rng, samples_per_edge)
            polylines[k] = (kind, idx, new)

    h_cuts = [line for kind, _i, line in polylines if kind == "h"]
    v_cuts = [line for kind, _i, line in polylines if kind == "v"]
    return PuzzleCuts(
        width=width,
        height=height,
        n_cols=n_cols,
        n_rows=n_rows,
        anchors=anchors,
        h_cuts=h_cuts,
        v_cuts=v_cuts,
        profile=profile,
    )


def render_cuts(cuts: PuzzleCuts, ax=None, show_anchors: bool = True):
    """Quick matplotlib visualization of a cut-set."""
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, cuts.width)
    ax.set_ylim(cuts.height, 0)
    ax.set_aspect("equal")

    # Border
    A = cuts.anchors
    for i in range(cuts.n_cols):
        ax.plot([A[0, i, 0], A[0, i + 1, 0]], [A[0, i, 1], A[0, i + 1, 1]], "k-", lw=1.6)
        ax.plot([A[-1, i, 0], A[-1, i + 1, 0]], [A[-1, i, 1], A[-1, i + 1, 1]], "k-", lw=1.6)
    for j in range(cuts.n_rows):
        ax.plot([A[j, 0, 0], A[j + 1, 0, 0]], [A[j, 0, 1], A[j + 1, 0, 1]], "k-", lw=1.6)
        ax.plot([A[j, -1, 0], A[j + 1, -1, 0]], [A[j, -1, 1], A[j + 1, -1, 1]], "k-", lw=1.6)

    for line in cuts.h_cuts:
        ax.plot(line[:, 0], line[:, 1], "k-", lw=1.2)
    for line in cuts.v_cuts:
        ax.plot(line[:, 0], line[:, 1], "k-", lw=1.2)

    if show_anchors:
        ax.scatter(A[:, :, 0].ravel(), A[:, :, 1].ravel(), c="red", s=14, zorder=5, alpha=0.7)
    ax.set_title(f"Jigsaw Cuts — {cuts.n_cols}x{cuts.n_rows}")
    return ax
