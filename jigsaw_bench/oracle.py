"""Greedy multi-cursor oracle for the jigsaw environment.

Solves a puzzle with *any* number of cursors ``c`` (up to :data:`MAX_CURSORS`)
and pieces ``p`` (up to :data:`MAX_PIECES`).  The oracle is a pure rule-based
:class:`~jigsaw_bench.JigsawModel` — call it ``(obs, step) -> {cid: ActionPoint}``.

Assignment
----------
At every reassignment moment the free cursors are matched to the available
(unsolved, unassigned) pieces by **greedy nearest pairing**:

* ``c > p`` — every piece gets its closest cursor; the leftover cursors are
  disemployed to the bottom-right corner.
* ``c == p`` — same greedy pairing; each cursor gets a unique piece.
* ``c < p`` — the closest ``c`` pieces are taken first; as cursors finish they
  pick up the next-closest remaining piece, until the puzzle is complete.

Movement
--------
**Approach** — each cursor aims at a guaranteed-interior point of its piece and
moves 1/3 of the *remaining* distance per step (or straight there within
``eps_px``).  It grabs on the step *after* it lands on the piece.

**Carry** — a target cursor position is computed so the held piece lands in its
solved pose; the cursor moves 1/3 of the remaining distance and rotates 1/3 of
the remaining (shortest-direction) angle each step.  Once the piece is within
the snap tolerance the cursor releases it and is reassigned to the next-closest
piece — or disemployed to the corner.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from .environment import ActionPoint, JigsawEnvironment

MAX_CURSORS: int = 32
MAX_PIECES: int = 500

_MOVE_FRACTION: float = 1.0 / 3.0   # fraction of remaining distance/angle per step
_CORNER_FRAC: tuple[float, float] = (0.97, 0.97)  # disemployed-cursor parking spot


def _safe_interior_local(sprite_rgba: np.ndarray) -> tuple[float, float]:
    """Return a sprite-local ``(x, y)`` pixel guaranteed to be deep inside the mask.

    Uses the alpha mask's Euclidean distance transform and picks the pixel
    farthest from any boundary (the most robustly-interior point), so a cursor
    aiming here is guaranteed to hit the piece even for concave tab/blank shapes.
    """
    alpha = sprite_rgba[..., 3] > 0
    if not alpha.any():
        h, w = alpha.shape
        return (w / 2.0, h / 2.0)
    # Pad with a background border so the sprite's crop edge counts as "outside";
    # the distance-transform maximum is then a genuinely interior pixel rather
    # than one sitting on the bounding-box boundary.
    padded = np.pad(alpha, 1, mode="constant", constant_values=False)
    dt = distance_transform_edt(padded)
    iy, ix = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return (float(ix - 1), float(iy - 1))


def _local_to_canvas(
    local_xy: tuple[float, float],
    sprite_centroid_local: tuple[float, float],
    centroid: tuple[float, float],
    rotation_deg: float,
) -> tuple[float, float]:
    """Map a sprite-local pixel to its canvas position under the piece's pose.

    Matches the environment's local→canvas convention:
    ``canvas = centroid + R(rot) · (local − sprite_centroid_local)``.
    """
    a = math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    lx = local_xy[0] - sprite_centroid_local[0]
    ly = local_xy[1] - sprite_centroid_local[1]
    return (centroid[0] + lx * ca - ly * sa,
            centroid[1] + lx * sa + ly * ca)


def _shortest_rot_err(rotation_deg: float) -> float:
    """Signed minimal angle (deg, in [−180, 180)) from ``rotation_deg`` to 0."""
    return ((rotation_deg + 180.0) % 360.0) - 180.0


def perimeter_cursor_starts(
    canvas_w: int, canvas_h: int, k: int, margin: float = 0.06,
) -> dict[int, tuple[float, float]]:
    """Spread ``k`` cursor start positions evenly around the canvas perimeter.

    Returns ``{cursor_id: (x, y)}`` in canvas pixels, inset from the edges by
    ``margin`` (a fraction of canvas size).  Distinct starts make each cursor's
    nearest piece well-defined from step 1 — useful when a policy must infer its
    target visually rather than from a piece assignment.
    """
    mx, my = canvas_w * margin, canvas_h * margin
    x0, y0, x1, y1 = mx, my, canvas_w - mx, canvas_h - my
    w, h = x1 - x0, y1 - y0
    perim = 2 * (w + h)
    starts: dict[int, tuple[float, float]] = {}
    for i in range(k):
        s = (i + 0.5) / k * perim
        if s < w:
            starts[i] = (x0 + s, y0)
        elif s < w + h:
            starts[i] = (x1, y0 + (s - w))
        elif s < 2 * w + h:
            starts[i] = (x1 - (s - w - h), y1)
        else:
            starts[i] = (x0, y1 - (s - 2 * w - h))
    return starts


class GreedyOracle:
    """Stateful greedy multi-cursor solver. Callable as a ``JigsawModel``."""

    def __init__(
        self,
        env: JigsawEnvironment,
        num_cursors: int | None = None,
        *,
        snap_pos_tol_px: float = 14.0,
        snap_rot_tol_deg: float = 8.0,
        eps_px: float = 2.0,
    ) -> None:
        n_pieces = len(env.pieces)
        if n_pieces > MAX_PIECES:
            raise ValueError(f"{n_pieces} pieces exceeds MAX_PIECES={MAX_PIECES}")

        if num_cursors is None or num_cursors <= 0:
            num_cursors = n_pieces
        self.K = min(num_cursors, MAX_CURSORS)

        self.env = env
        self.snap_pos_tol_px = snap_pos_tol_px
        self.snap_rot_tol_deg = snap_rot_tol_deg
        self.eps_px = eps_px

        # Precompute per-piece constants.
        self.piece_indices: list[int] = sorted(env.pieces.keys())
        self.interior_local: dict[int, tuple[float, float]] = {}
        self.sprite_centroid_local: dict[int, tuple[float, float]] = {}
        for idx, ps in env.pieces.items():
            self.interior_local[idx] = _safe_interior_local(ps.sprite_rgba)
            self.sprite_centroid_local[idx] = ps.sprite_centroid_local
        self.targets: dict[int, tuple[float, float]] = env.target_centroids()

        # Mutable oracle state.
        self.solved: set[int] = set()
        self.assigned: dict[int, int] = {}                       # piece -> cursor
        self.cursor: dict[int, dict] = {                         # per-cursor FSM
            cid: {"piece": None, "phase": "free"} for cid in range(self.K)
        }
        self.cursor_pos: dict[int, tuple[float, float]] = {
            cid: (0.0, 0.0) for cid in range(self.K)
        }

    # ---------- geometry helpers ----------

    def _interior_canvas(self, idx: int) -> tuple[float, float]:
        ps = self.env.pieces[idx]
        return _local_to_canvas(
            self.interior_local[idx], self.sprite_centroid_local[idx],
            ps.centroid, ps.rotation_deg,
        )

    # ---------- assignment ----------

    def _reassign(self) -> None:
        """Greedily match every free cursor to its nearest available piece."""
        free = [
            cid for cid in range(self.K)
            if self.cursor[cid]["piece"] is None and self.cursor[cid]["phase"] != "idle"
        ]
        if not free:
            return

        assigned_pieces = set(self.assigned.keys())
        avail = [
            p for p in self.piece_indices
            if p not in self.solved and p not in assigned_pieces
        ]

        # Greedy nearest pairing.
        while free and avail:
            best: tuple[float, int, int] | None = None
            for cid in free:
                px, py = self.cursor_pos[cid]
                for p in avail:
                    qx, qy = self._interior_canvas(p)
                    d2 = (px - qx) ** 2 + (py - qy) ** 2
                    if best is None or d2 < best[0]:
                        best = (d2, cid, p)
            assert best is not None
            _, cid, p = best
            self.assigned[p] = cid
            self.cursor[cid] = {"piece": p, "phase": "approach"}
            free.remove(cid)
            avail.remove(p)

        # Any cursor with no piece to chase is disemployed.
        for cid in free:
            self.cursor[cid]["phase"] = "idle"

    # ---------- main policy ----------

    def __call__(self, obs: np.ndarray, step: int) -> dict[int, ActionPoint]:
        env = self.env
        W, H = env.canvas_w, env.canvas_h
        corner = (W * _CORNER_FRAC[0], H * _CORNER_FRAC[1])
        priority_scale = 1.0 / max(1, self.K - 1)

        # Sync tracked positions with the environment's truth.
        for cid, cs in env.cursors.items():
            if cid < self.K:
                self.cursor_pos[cid] = (cs.last_x, cs.last_y)

        self._reassign()

        actions: dict[int, ActionPoint] = {}
        for cid in range(self.K):
            st = self.cursor[cid]
            phase = st["phase"]
            pri = cid * priority_scale

            if phase == "idle":
                actions[cid] = ActionPoint(
                    corner[0] / W, corner[1] / H, grab=False, finished=True,
                )
                continue

            piece = st["piece"]

            # If we believe we're carrying but the env says we don't hold the
            # intended piece, the grab was blocked — fall back to re-approach.
            if phase == "carry":
                cs = env.cursors.get(cid)
                if cs is None or cs.held_piece != piece:
                    phase = st["phase"] = "approach"

            if phase == "approach":
                px, py = self.cursor_pos[cid]
                tx, ty = self._interior_canvas(piece)
                if math.hypot(tx - px, ty - py) <= self.eps_px:
                    # Landed on the piece; grab on the next step.
                    nx, ny = tx, ty
                    st["phase"] = "grab"
                else:
                    nx = px + (tx - px) * _MOVE_FRACTION
                    ny = py + (ty - py) * _MOVE_FRACTION
                self.cursor_pos[cid] = (nx, ny)
                actions[cid] = ActionPoint(nx / W, ny / H, grab=False, render_priority=pri)

            elif phase == "grab":
                tx, ty = self._interior_canvas(piece)   # piece hasn't moved
                self.cursor_pos[cid] = (tx, ty)
                st["phase"] = "carry"
                actions[cid] = ActionPoint(tx / W, ty / H, grab=True, render_priority=pri)

            else:  # phase == "carry"
                ps = env.pieces[piece]
                cx, cy = ps.centroid
                txp, typ = self.targets[piece]
                rot_err = _shortest_rot_err(ps.rotation_deg)

                # Within snap tolerance → release and free the cursor.
                if (math.hypot(cx - txp, cy - typ) <= self.snap_pos_tol_px
                        and abs(rot_err) <= self.snap_rot_tol_deg):
                    px, py = self.cursor_pos[cid]
                    actions[cid] = ActionPoint(px / W, py / H, grab=False, render_priority=pri)
                    self.solved.add(piece)
                    self.assigned.pop(piece, None)
                    self.cursor[cid] = {"piece": None, "phase": "free"}
                    continue

                # Move the grab point toward where the piece lands solved (rot 0),
                # rotating a third of the remaining (shortest-direction) angle.
                cs = env.cursors.get(cid)
                offx, offy = cs.grab_offset_local if (cs and cs.grab_offset_local) else (0.0, 0.0)
                target_px, target_py = txp + offx, typ + offy
                px, py = self.cursor_pos[cid]
                if math.hypot(target_px - px, target_py - py) <= self.eps_px:
                    nx, ny = target_px, target_py
                else:
                    nx = px + (target_px - px) * _MOVE_FRACTION
                    ny = py + (target_py - py) * _MOVE_FRACTION
                self.cursor_pos[cid] = (nx, ny)

                step_deg = (-rot_err) * _MOVE_FRACTION
                rot_norm = float(np.clip(step_deg / env.max_step_rotation_deg, -1.0, 1.0))
                actions[cid] = ActionPoint(
                    nx / W, ny / H, grab=True,
                    rotation_delta=rot_norm, render_priority=pri,
                )

        return actions


def make_greedy_oracle(
    env: JigsawEnvironment,
    num_cursors: int | None = None,
    *,
    snap_pos_tol_px: float = 14.0,
    snap_rot_tol_deg: float = 8.0,
) -> GreedyOracle:
    """Convenience constructor for :class:`GreedyOracle`.

    ``num_cursors=None`` uses one cursor per piece (capped at
    :data:`MAX_CURSORS`).  Pass an integer for any other cursor count.
    """
    return GreedyOracle(
        env, num_cursors,
        snap_pos_tol_px=snap_pos_tol_px,
        snap_rot_tol_deg=snap_rot_tol_deg,
    )
