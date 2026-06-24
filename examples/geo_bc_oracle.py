"""Behavioral cloning oracle for the geometric (structured-input) model interface.

Demonstrates the full pipeline for training and evaluating a learned model that
operates on geometric observations rather than raw pixels:

1. **Collect** oracle trajectories as (features, action) pairs.
2. **Train** a two-layer MLP via supervised regression (behavioral cloning).
3. **Deploy** the trained model as a standard :class:`~jigsaw_bench.JigsawModel`.
4. **Evaluate** against the benchmark and compare with the rule-based oracle.

The model is intentionally framework-free: all training and inference runs on
plain NumPy so the example has no extra dependencies beyond those already
required by ``jigsaw_bench``.

Geometric I/O schema
--------------------
Each cursor managing piece *k* is described by a **13-dimensional feature vector**:

    col  feature
    ---  -------
    0    cursor_x / cw          cursor position x (normalised)
    1    cursor_y / ch          cursor position y
    2    is_holding             1.0 if cursor currently holds a piece
    3    piece_cx / cw          target piece centroid x
    4    piece_cy / ch          target piece centroid y
    5    sin(piece_rot)         rotation as unit-circle coords
    6    cos(piece_rot)
    7    target_x / cw          target (solved) centroid x
    8    target_y / ch          target (solved) centroid y
    9    Δcursor_x / cw         (piece_cx − cursor_x) / cw
    10   Δcursor_y / ch         (piece_cy − cursor_y) / ch
    11   Δtarget_x / cw         (target_x − piece_cx) / cw
    12   Δtarget_y / ch         (target_y − piece_cy) / ch

The model predicts a **4-dimensional action vector** per cursor:

    col  field            meaning
    ---  -----            -------
    0    action_x         target cursor x (normalised), i.e. ActionPoint.x
    1    action_y         target cursor y (normalised), i.e. ActionPoint.y
    2    grab_logit       sigmoid > 0.5 → grab=True (raw logit during training)
    3    rotation_delta   rotation correction, clipped to (−1, 1)

Usage
-----
    # collect 8 oracle rollouts, train, then benchmark:
    python examples/geo_bc_oracle.py path/to/image.jpg --rollouts 8

    # skip training and just run the rule-based oracle for comparison:
    python examples/geo_bc_oracle.py path/to/image.jpg --oracle-only
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from jigsaw_bench import (
    ActionPoint,
    JigsawEnvironment,
    benchmark_model,
    generate_puzzle,
    shuffle_pieces,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BC_FEAT_DIM: int = 13   # per-cursor feature vector dimension (see module docstring)
BC_ACT_DIM: int = 4     # (action_x, action_y, grab_logit, rotation_delta)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def cursor_features(
    env: JigsawEnvironment,
    cursor_x: float,
    cursor_y: float,
    is_holding: bool,
    piece_idx: int,
) -> np.ndarray:
    """Build a ``(BC_FEAT_DIM,)`` float32 feature vector for a single (cursor, piece) pair.

    Parameters
    ----------
    env:
        Live environment (used for canvas size + current piece state).
    cursor_x, cursor_y:
        Current cursor position in *canvas pixels* (not normalised).
    is_holding:
        Whether this cursor is currently holding the piece.
    piece_idx:
        The piece this cursor is assigned to solve.
    """
    cw, ch = env.canvas_w, env.canvas_h
    centroids = env.piece_centroids()
    targets = env.target_centroids()
    rotations = env.piece_rotations()

    pcx, pcy = centroids[piece_idx]
    tx, ty = targets[piece_idx]
    rot = rotations[piece_idx]

    cx_n = cursor_x / cw
    cy_n = cursor_y / ch

    return np.array(
        [
            cx_n,
            cy_n,
            float(is_holding),
            pcx / cw,
            pcy / ch,
            math.sin(math.radians(rot)),
            math.cos(math.radians(rot)),
            tx / cw,
            ty / ch,
            (pcx - cursor_x) / cw,
            (pcy - cursor_y) / ch,
            (tx - pcx) / cw,
            (ty - pcy) / ch,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Minimal numpy-only MLP
# ---------------------------------------------------------------------------

class _MLP:
    """Two-layer ReLU MLP trained with mini-batch SGD.

    Intentionally avoids any deep-learning framework so the example works with
    the base ``jigsaw_bench`` dependencies.
    """

    def __init__(
        self,
        in_dim: int = BC_FEAT_DIM,
        hidden: int = 128,
        out_dim: int = BC_ACT_DIM,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        s1 = math.sqrt(2.0 / in_dim)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = (rng.standard_normal((in_dim, hidden)) * s1).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, out_dim)) * s2).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass; ``x`` can be (feat_dim,) or (batch, feat_dim)."""
        batched = x.ndim == 2
        if not batched:
            x = x[None]
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out if batched else out[0]

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        lr: float = 3e-3,
        epochs: int = 300,
        batch_size: int = 512,
        verbose: bool = True,
    ) -> list[float]:
        """Train in-place with MSE loss + sigmoid BCE on the grab logit column.

        Returns per-epoch loss values.
        """
        n = len(X)
        losses: list[float] = []
        for ep in range(1, epochs + 1):
            perm = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                xb = X[perm[start : start + batch_size]]
                yb = Y[perm[start : start + batch_size]]
                bs = len(xb)

                # Forward
                h_pre = xb @ self.W1 + self.b1          # (bs, H)
                h = np.maximum(0.0, h_pre)               # ReLU
                out = h @ self.W2 + self.b2              # (bs, 4)

                # MSE on all dims (grab column uses raw logit target ±3)
                diff = out - yb
                loss = float((diff ** 2).mean())
                epoch_loss += loss
                n_batches += 1

                # Backward (MSE gradient)
                dout = 2.0 * diff / (bs * BC_ACT_DIM)
                dW2 = h.T @ dout
                db2 = dout.sum(0)
                dh = dout @ self.W2.T
                dh_pre = dh * (h_pre > 0)
                dW1 = xb.T @ dh_pre
                db1 = dh_pre.sum(0)

                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2

            avg = epoch_loss / max(1, n_batches)
            losses.append(avg)
            if verbose and ep % 50 == 0:
                print(f"  epoch {ep:4d}/{epochs}  loss={avg:.5f}")
        return losses


# ---------------------------------------------------------------------------
# Oracle data collector
# ---------------------------------------------------------------------------

def collect_oracle_data(
    puzzle_factory: Callable[[], tuple],
    *,
    n_rollouts: int = 8,
    num_cursors: int = 4,
    max_steps: int = 3000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the rule-based oracle and record (features, action) pairs.

    Parameters
    ----------
    puzzle_factory:
        No-arg callable that returns ``(puzzle, layout)``; called once per rollout
        so each rollout can use a different random seed.
    n_rollouts:
        Number of independent puzzle rollouts to collect from.
    num_cursors:
        Number of parallel cursors passed to the oracle.
    max_steps:
        Hard step limit per rollout.
    seed:
        Base RNG seed; each rollout increments it by 1.

    Returns
    -------
    X : (M, BC_FEAT_DIM) float32
    Y : (M, BC_ACT_DIM) float32
        Grab column (index 2) is encoded as +3.0 for grab=True, −3.0 for grab=False
        so it sits well inside the linear regime before sigmoid and MSE loss still
        works as an approximation of BCE.
    """
    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []

    for rollout in range(n_rollouts):
        puzzle, layout = puzzle_factory()
        env = JigsawEnvironment(puzzle, layout)
        env.reset()

        oracle, _state = _make_oracle_with_state(env, num_cursors=num_cursors)

        for step in range(1, max_steps + 1):
            # Record (features, action) BEFORE stepping
            for cid, st in _state.items():
                if st["piece"] is None:
                    continue
                piece_idx = st["piece"]
                cs = env.cursors.get(cid)
                curs_x = cs.last_x if cs is not None else 0.0
                curs_y = cs.last_y if cs is not None else 0.0
                is_holding = cs is not None and cs.held_piece is not None

                feats = cursor_features(env, curs_x, curs_y, is_holding, piece_idx)

                # Get the oracle action for this cursor
                actions = oracle(env.render(), step)
                ap = actions.get(cid)
                if ap is None:
                    continue

                act = np.array(
                    [
                        ap.x,                                  # already normalised
                        ap.y,
                        3.0 if ap.grab else -3.0,             # logit encoding
                        float(np.clip(ap.rotation_delta, -1.0, 1.0)),
                    ],
                    dtype=np.float32,
                )
                X_parts.append(feats)
                Y_parts.append(act)

            env.step(oracle(env.render(), step))
            if env.is_solved():
                break

        print(
            f"  rollout {rollout + 1}/{n_rollouts} — {len(X_parts)} samples so far"
        )

    X = np.stack(X_parts) if X_parts else np.zeros((0, BC_FEAT_DIM), dtype=np.float32)
    Y = np.stack(Y_parts) if Y_parts else np.zeros((0, BC_ACT_DIM), dtype=np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Oracle state machine (shared between collection and rule-based evaluation)
# ---------------------------------------------------------------------------

def _make_oracle_with_state(env: JigsawEnvironment, num_cursors: int = 4):
    """Return (model_fn, state_dict) for a rule-based oracle.

    The state dict is *mutable* and shared with the caller so the data collector
    can inspect which piece each cursor is currently targeting.
    """
    targets = env.target_centroids()
    queue = list(targets.keys())
    state: dict[int, dict] = {
        cid: {"piece": None, "phase": "pick"} for cid in range(num_cursors)
    }

    def model(obs: np.ndarray, step: int) -> dict[int, ActionPoint]:
        actions: dict[int, ActionPoint] = {}
        cw, ch = env.canvas_w, env.canvas_h
        centroids = env.piece_centroids()
        rotations = env.piece_rotations()

        for cid, st in state.items():
            if st["piece"] is None:
                if not queue:
                    actions[cid] = ActionPoint(0.5, 0.5, grab=False, finished=True)
                    continue
                st["piece"] = queue.pop()
                st["phase"] = "approach"

            idx = st["piece"]
            cx, cy = centroids[idx]
            tx, ty = targets[idx]
            cur_rot = rotations[idx]
            rot_err = ((cur_rot + 180) % 360) - 180

            if st["phase"] == "approach":
                actions[cid] = ActionPoint(
                    cx / cw, cy / ch,
                    grab=True,
                    render_priority=cid / max(1, num_cursors - 1),
                )
                st["phase"] = "carry"
            elif st["phase"] == "carry":
                dx, dy = tx - cx, ty - cy
                dist = math.hypot(dx, dy)
                if dist > 1.0:
                    factor = min(40.0, dist) / dist
                    nx, ny = cx + dx * factor, cy + dy * factor
                else:
                    nx, ny = tx, ty
                rot_norm = float(np.clip(-rot_err / 180.0, -1.0, 1.0)) * 0.2
                actions[cid] = ActionPoint(
                    nx / cw, ny / ch,
                    grab=True,
                    rotation_delta=rot_norm,
                    render_priority=cid / max(1, num_cursors - 1),
                )
                if dist < 0.5 and abs(rot_err) < 0.5:
                    st["phase"] = "release"
            elif st["phase"] == "release":
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                st["phase"] = "pick"

        return actions

    return model, state


def make_rule_oracle(env: JigsawEnvironment, num_cursors: int = 4) -> Callable:
    """Rule-based oracle as a bare :class:`~jigsaw_bench.JigsawModel` callable."""
    model, _ = _make_oracle_with_state(env, num_cursors=num_cursors)
    return model


# ---------------------------------------------------------------------------
# BC model inference wrapper
# ---------------------------------------------------------------------------

def make_bc_oracle(
    mlp: _MLP,
    env: JigsawEnvironment,
    num_cursors: int = 4,
) -> Callable:
    """Wrap a trained :class:`_MLP` inside the oracle's piece-assignment state machine.

    The MLP replaces the explicit rule-based action formulas with learned
    regression — piece assignment and queueing logic is unchanged.

    Returns a :class:`~jigsaw_bench.JigsawModel`-compatible callable.
    """
    targets = env.target_centroids()
    queue = list(targets.keys())
    state: dict[int, dict] = {
        cid: {"piece": None, "phase": "pick"} for cid in range(num_cursors)
    }

    def model(obs: np.ndarray, step: int) -> dict[int, ActionPoint]:
        actions: dict[int, ActionPoint] = {}
        cw, ch = env.canvas_w, env.canvas_h

        for cid, st in state.items():
            if st["piece"] is None:
                if not queue:
                    actions[cid] = ActionPoint(0.5, 0.5, grab=False, finished=True)
                    continue
                st["piece"] = queue.pop()

            idx = st["piece"]
            cs = env.cursors.get(cid)
            curs_x = cs.last_x if cs is not None else 0.0
            curs_y = cs.last_y if cs is not None else 0.0
            is_holding = cs is not None and cs.held_piece is not None

            feats = cursor_features(env, curs_x, curs_y, is_holding, idx)
            pred = mlp.forward(feats)

            ax = float(np.clip(pred[0], 0.0, 1.0))
            ay = float(np.clip(pred[1], 0.0, 1.0))
            grab = bool(pred[2] > 0.0)
            rot_delta = float(np.clip(pred[3], -1.0, 1.0))

            # Retire piece once it is delivered (position + rotation within snap threshold)
            centroids = env.piece_centroids()
            tx, ty = targets[idx]
            pcx, pcy = centroids[idx]
            rotations = env.piece_rotations()
            rot_err = ((rotations[idx] + 180) % 360) - 180
            if math.hypot(pcx - tx, pcy - ty) < 2.0 and abs(rot_err) < 2.0:
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                continue

            actions[cid] = ActionPoint(
                ax, ay,
                grab=grab,
                rotation_delta=rot_delta,
                render_priority=cid / max(1, num_cursors - 1),
            )

        return actions

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a behavioral cloning oracle from geometric observations."
    )
    parser.add_argument("image", type=Path, help="Source image path.")
    parser.add_argument("--out", type=Path, default=Path("out"),
                        help="Output directory for rendered frames.")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cursors", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=8,
                        help="Oracle rollouts to collect training data from.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--oracle-only", action="store_true",
                        help="Skip training; only run the rule-based oracle for comparison.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # --- shared puzzle factory -------------------------------------------
    def _make_env(seed: int):
        puzzle = generate_puzzle(
            str(args.image),
            width=900, height=600,
            n_cols=args.cols, n_rows=args.rows,
            seed=seed,
        )
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=seed)
        return puzzle, layout

    # --- rule-based oracle (baseline) ------------------------------------
    print("\n=== Rule-based oracle (baseline) ===")
    puzzle, layout = _make_env(args.seed)
    env_rb = JigsawEnvironment(puzzle, layout)
    rb_model = make_rule_oracle(env_rb, num_cursors=args.cursors)
    rb_result = benchmark_model(
        env_rb, rb_model,
        max_steps=3000, snap_to=True,
        snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
        record_history=False,
    )
    print(rb_result.summary())
    Image.fromarray(env_rb.render()).save(args.out / "oracle_final.png")

    if args.oracle_only:
        return

    # --- collect behavioral cloning data ---------------------------------
    print(f"\n=== Collecting data from {args.rollouts} oracle rollouts ===")

    def _factory():
        return _make_env(args.seed)

    X, Y = collect_oracle_data(
        _factory,
        n_rollouts=args.rollouts,
        num_cursors=args.cursors,
        seed=args.seed,
    )
    print(f"Collected {len(X)} training samples  (features={X.shape[1]}, actions={Y.shape[1]})")

    if len(X) == 0:
        print("No samples collected — aborting training.")
        return

    # Normalise features (zero mean, unit variance per column)
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True) + 1e-6
    X_norm = (X - mu) / sigma

    # --- train BC model --------------------------------------------------
    print(f"\n=== Training BC MLP (hidden={args.hidden}, epochs={args.epochs}) ===")
    mlp = _MLP(in_dim=BC_FEAT_DIM, hidden=args.hidden, out_dim=BC_ACT_DIM, seed=args.seed)
    mlp.fit(X_norm, Y, lr=3e-3, epochs=args.epochs, batch_size=512, verbose=True)

    # Wrap the model so it normalises at inference time
    _mu, _sigma = mu[0], sigma[0]

    class _NormMLP(_MLP):
        def forward(self, x: np.ndarray) -> np.ndarray:
            return mlp.forward((x - _mu) / _sigma)

    norm_mlp = _NormMLP.__new__(_NormMLP)
    norm_mlp.W1, norm_mlp.b1 = mlp.W1, mlp.b1
    norm_mlp.W2, norm_mlp.b2 = mlp.W2, mlp.b2

    # --- evaluate BC model -----------------------------------------------
    print("\n=== Evaluating BC oracle ===")
    puzzle, layout = _make_env(args.seed)
    env_bc = JigsawEnvironment(puzzle, layout)
    bc_model = make_bc_oracle(norm_mlp, env_bc, num_cursors=args.cursors)
    bc_result = benchmark_model(
        env_bc, bc_model,
        max_steps=3000, snap_to=True,
        snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
        record_history=False,
    )
    print(bc_result.summary())
    Image.fromarray(env_bc.render()).save(args.out / "bc_final.png")

    print("\n=== Summary ===")
    print(f"  Rule-based oracle  piecewise_score={rb_result.piecewise_score:.4f}  steps={rb_result.steps}")
    print(f"  BC oracle          piecewise_score={bc_result.piecewise_score:.4f}  steps={bc_result.steps}")


if __name__ == "__main__":
    main()
