"""Behavioral cloning oracle for the geometric (frame + cursor) model interface.

Demonstrates the full pipeline:

1. **Collect** oracle trajectories as (image_frame, cursor_state, action) triples.
2. **Train** a two-layer MLP via supervised regression (behavioral cloning).
3. **Deploy** the trained model as a standard :class:`~jigsaw_bench.JigsawModel`.
4. **Evaluate** against the benchmark, with an optional comparison to the rule-based oracle.

Framework-free: all training and inference uses plain NumPy.

Model I/O schema
----------------
The model receives two inputs **per cursor per step**:

    frame_small  (BC_FRAME_H, BC_FRAME_W, 3) float32 normalised to [0, 1]
                 — downsampled version of the rendered canvas
    cursor_feat  (GEO_CURSOR_DIM,) float32
                 — [x/W, y/H, is_holding, held_cx/W, held_cy/H]

These are concatenated into a single ``(BC_FEAT_DIM,)`` vector.  The model
predicts a ``(BC_ACT_DIM,)`` action vector:

    col  field            meaning
    ---  -----            -------
    0    action_x         target cursor x (normalised 0–1)
    1    action_y         target cursor y
    2    grab_logit        > 0 → grab=True (raw logit; sigmoid applied at inference)
    3    rotation_delta    clipped to (−1, 1) → (−180°, +180°) per step

Piece coordinates and target positions are intentionally **not** given to the
model — it must infer them from the rendered frame, just as a human would.

Simultaneous movement
---------------------
By default ``num_cursors=None``, which sets K = N (one cursor per piece).
Every piece moves in parallel from the very first step.  Pass an integer to
cap parallelism; pieces are distributed round-robin and cursors pick the next
free piece from a shared queue when they finish.

Usage
-----
    # full pipeline: collect, train, evaluate
    python examples/geo_bc_oracle.py path/to/image.jpg --rollouts 8

    # skip training, benchmark only the rule-based oracle
    python examples/geo_bc_oracle.py path/to/image.jpg --oracle-only

    # use all cursors (one per piece) for maximum parallelism
    python examples/geo_bc_oracle.py path/to/image.jpg --cursors 0
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
from jigsaw_bench.geo_model import GEO_CURSOR_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BC_FRAME_H: int = 32
BC_FRAME_W: int = 32
BC_FRAME_FLAT: int = BC_FRAME_H * BC_FRAME_W * 3   # 3 072
BC_FEAT_DIM: int = BC_FRAME_FLAT + GEO_CURSOR_DIM   # 3 077
BC_ACT_DIM: int = 4  # (action_x, action_y, grab_logit, rotation_delta)


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _downsample_frame(frame: np.ndarray) -> np.ndarray:
    """Resize an (H, W, 3) uint8 frame → (BC_FRAME_H, BC_FRAME_W, 3) float32 in [0,1]."""
    img = Image.fromarray(frame).resize((BC_FRAME_W, BC_FRAME_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def build_cursor_feat(env: JigsawEnvironment, cursor_id: int) -> np.ndarray:
    """Extract a ``(GEO_CURSOR_DIM,)`` feature vector for *cursor_id*.

    This is exactly the cursor row from :func:`~jigsaw_bench.geo_observation`,
    exposed as a standalone helper for training pipelines that loop over cursors.
    """
    cw, ch = env.canvas_w, env.canvas_h
    cs = env.cursors.get(cursor_id)
    if cs is None:
        return np.zeros(GEO_CURSOR_DIM, dtype=np.float32)
    centroids = env.piece_centroids()
    held_cx, held_cy, is_holding = 0.0, 0.0, 0.0
    if cs.held_piece is not None and cs.held_piece in centroids:
        hcx, hcy = centroids[cs.held_piece]
        held_cx, held_cy = hcx / cw, hcy / ch
        is_holding = 1.0
    return np.array(
        [cs.last_x / cw, cs.last_y / ch, is_holding, held_cx, held_cy],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Minimal numpy-only MLP
# ---------------------------------------------------------------------------

class _MLP:
    """Two-layer ReLU MLP trained with mini-batch SGD.

    No deep-learning framework required — uses only NumPy so the example
    works with the base ``jigsaw_bench`` dependencies.
    """

    def __init__(
        self,
        in_dim: int = BC_FEAT_DIM,
        hidden: int = 256,
        out_dim: int = BC_ACT_DIM,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden)) * math.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, out_dim)) * math.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass; ``x`` can be ``(feat_dim,)`` or ``(batch, feat_dim)``."""
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
        lr: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list[float]:
        """Train in-place with MSE loss.  Returns per-epoch mean losses."""
        n = len(X)
        losses: list[float] = []
        rng = np.random.default_rng(0)
        for ep in range(1, epochs + 1):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                xb = X[perm[start : start + batch_size]]
                yb = Y[perm[start : start + batch_size]]
                bs = len(xb)

                h_pre = xb @ self.W1 + self.b1
                h = np.maximum(0.0, h_pre)
                out = h @ self.W2 + self.b2

                diff = out - yb
                loss = float((diff ** 2).mean())
                epoch_loss += loss
                n_batches += 1

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
# Oracle state machine (simultaneous, variable K)
# ---------------------------------------------------------------------------

def _make_oracle(env: JigsawEnvironment, num_cursors: int | None = None):
    """Return an oracle model (and its mutable state dict).

    Parameters
    ----------
    num_cursors:
        Number of simultaneous cursors.  ``None`` (default) sets K = N so
        every piece moves in parallel.  Pass an integer to cap parallelism;
        each cursor draws from a shared FIFO queue when it finishes a piece.
    """
    N = len(env.pieces)
    K = N if (num_cursors is None or num_cursors <= 0) else min(num_cursors, N)

    targets = env.target_centroids()
    queue: list[int] = list(targets.keys())   # pieces not yet assigned
    # state per cursor: which piece, which phase
    state: dict[int, dict] = {cid: {"piece": None, "phase": "pick"} for cid in range(K)}

    def model(obs: np.ndarray, step: int) -> dict[int, ActionPoint]:
        actions: dict[int, ActionPoint] = {}
        cw, ch = env.canvas_w, env.canvas_h
        centroids = env.piece_centroids()
        rotations = env.piece_rotations()
        render_scale = 1.0 / max(1, K - 1) if K > 1 else 1.0

        for cid, st in state.items():
            if st["piece"] is None:
                if not queue:
                    actions[cid] = ActionPoint(0.5, 0.5, grab=False, finished=True)
                    continue
                st["piece"] = queue.pop(0)
                st["phase"] = "approach"

            idx = st["piece"]
            cx, cy = centroids[idx]
            tx, ty = targets[idx]
            rot_err = ((rotations[idx] + 180) % 360) - 180
            pri = cid * render_scale

            if st["phase"] == "approach":
                actions[cid] = ActionPoint(cx / cw, cy / ch, grab=True, render_priority=pri)
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
                    nx / cw, ny / ch, grab=True,
                    rotation_delta=rot_norm, render_priority=pri,
                )
                if dist < 0.5 and abs(rot_err) < 0.5:
                    st["phase"] = "release"

            elif st["phase"] == "release":
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                st["phase"] = "pick"

        return actions

    return model, state


def make_rule_oracle(env: JigsawEnvironment, num_cursors: int | None = None) -> Callable:
    """Rule-based oracle as a bare :class:`~jigsaw_bench.JigsawModel` callable.

    Parameters
    ----------
    num_cursors:
        ``None`` → one cursor per piece (maximum parallelism).
        Integer → that many simultaneous cursors sharing a piece queue.
    """
    model, _ = _make_oracle(env, num_cursors=num_cursors)
    return model


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_oracle_data(
    puzzle_factory: Callable[[], tuple],
    *,
    n_rollouts: int = 8,
    num_cursors: int | None = None,
    max_steps: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the rule-based oracle and record ``(features, action)`` pairs.

    Features are ``(BC_FEAT_DIM,)`` = downsampled-frame-flat + cursor-state.
    Actions are ``(BC_ACT_DIM,)`` = (action_x, action_y, grab_logit, rot_delta).

    Parameters
    ----------
    puzzle_factory:
        No-arg callable returning ``(puzzle, layout)``.
    n_rollouts:
        Number of independent rollouts.
    num_cursors:
        Cursor count per rollout (``None`` = one per piece).
    max_steps:
        Hard step limit per rollout.

    Returns
    -------
    X : (M, BC_FEAT_DIM) float32
    Y : (M, BC_ACT_DIM) float32
        Grab column (index 2) encoded as +3 / −3 (raw logit targets).
    """
    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []

    for rollout in range(n_rollouts):
        puzzle, layout = puzzle_factory()
        env = JigsawEnvironment(puzzle, layout)
        env.reset()

        oracle, state = _make_oracle(env, num_cursors=num_cursors)

        for step in range(1, max_steps + 1):
            # Render once — both data-collection and env.step use this frame
            frame = env.render()
            frame_small = _downsample_frame(frame)
            frame_flat = frame_small.ravel()   # (BC_FRAME_FLAT,)

            # Get oracle actions (advances the oracle's state machine once)
            actions = oracle(frame, step)

            # Record one sample per active cursor
            for cid, ap in actions.items():
                if ap.finished:
                    continue
                cursor_feat = build_cursor_feat(env, cid)
                feat = np.concatenate([frame_flat, cursor_feat])  # (BC_FEAT_DIM,)
                act = np.array(
                    [
                        ap.x,
                        ap.y,
                        3.0 if ap.grab else -3.0,
                        float(np.clip(ap.rotation_delta, -1.0, 1.0)),
                    ],
                    dtype=np.float32,
                )
                X_parts.append(feat)
                Y_parts.append(act)

            env.step(actions)
            if env.is_solved():
                break

        print(f"  rollout {rollout + 1}/{n_rollouts} — {len(X_parts)} samples collected")

    X = np.stack(X_parts) if X_parts else np.zeros((0, BC_FEAT_DIM), dtype=np.float32)
    Y = np.stack(Y_parts) if Y_parts else np.zeros((0, BC_ACT_DIM), dtype=np.float32)
    return X, Y


# ---------------------------------------------------------------------------
# BC model inference wrapper
# ---------------------------------------------------------------------------

def make_bc_oracle(
    mlp: _MLP,
    env: JigsawEnvironment,
    num_cursors: int | None = None,
    frame_mu: np.ndarray | None = None,
    frame_sigma: np.ndarray | None = None,
) -> Callable:
    """Wrap a trained :class:`_MLP` in the oracle's piece-assignment state machine.

    The MLP sees ``(downsampled_frame_flat + cursor_feat)`` and predicts
    ``(action_x, action_y, grab_logit, rotation_delta)``.  Piece coordinates
    and targets are unknown to the MLP — it infers them from the frame.

    Parameters
    ----------
    mlp:
        Trained MLP.
    env:
        The environment to run against (used for K and canvas size).
    num_cursors:
        ``None`` → one cursor per piece.  Integer → cap parallelism.
    frame_mu, frame_sigma:
        Per-feature normalisation statistics from training.  If provided,
        features are standardised before MLP forward pass.
    """
    N = len(env.pieces)
    K = N if (num_cursors is None or num_cursors <= 0) else min(num_cursors, N)

    targets = env.target_centroids()
    queue: list[int] = list(targets.keys())
    state: dict[int, dict] = {cid: {"piece": None, "phase": "pick"} for cid in range(K)}

    def model(obs: np.ndarray, step: int) -> dict[int, ActionPoint]:
        actions: dict[int, ActionPoint] = {}
        cw, ch = env.canvas_w, env.canvas_h
        centroids = env.piece_centroids()
        rotations = env.piece_rotations()

        # Compute downsampled frame once and reuse across all cursors
        frame_flat = _downsample_frame(obs).ravel()

        render_scale = 1.0 / max(1, K - 1) if K > 1 else 1.0

        for cid, st in state.items():
            if st["piece"] is None:
                if not queue:
                    actions[cid] = ActionPoint(0.5, 0.5, grab=False, finished=True)
                    continue
                st["piece"] = queue.pop(0)

            idx = st["piece"]
            cursor_feat = build_cursor_feat(env, cid)
            feat = np.concatenate([frame_flat, cursor_feat])
            if frame_mu is not None and frame_sigma is not None:
                feat = (feat - frame_mu) / frame_sigma

            pred = mlp.forward(feat)
            ax = float(np.clip(pred[0], 0.0, 1.0))
            ay = float(np.clip(pred[1], 0.0, 1.0))
            grab = bool(pred[2] > 0.0)
            rot_delta = float(np.clip(pred[3], -1.0, 1.0))

            # Retire piece once it lands within snap distance
            tx, ty = targets[idx]
            pcx, pcy = centroids[idx]
            rot_err = ((rotations[idx] + 180) % 360) - 180
            if math.hypot(pcx - tx, pcy - ty) < 2.0 and abs(rot_err) < 2.0:
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                continue

            actions[cid] = ActionPoint(
                ax, ay, grab=grab,
                rotation_delta=rot_delta,
                render_priority=cid * render_scale,
            )

        return actions

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a behavioral cloning oracle from frame + cursor observations."
    )
    parser.add_argument("image", type=Path, help="Source image path.")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument(
        "--cursors", type=int, default=0,
        help="Number of simultaneous cursors (0 or omit = one per piece).",
    )
    parser.add_argument("--rollouts", type=int, default=8,
                        help="Oracle rollouts for training data collection.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--oracle-only", action="store_true",
                        help="Skip BC training; only run the rule-based oracle.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    num_cursors = None if args.cursors <= 0 else args.cursors

    def _make_env(seed: int):
        puzzle = generate_puzzle(
            str(args.image), width=900, height=600,
            n_cols=args.cols, n_rows=args.rows, seed=seed,
        )
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=seed)
        return puzzle, layout

    # --- rule-based oracle (baseline) ---
    print("\n=== Rule-based oracle ===")
    puzzle, layout = _make_env(args.seed)
    env_rb = JigsawEnvironment(puzzle, layout)
    rb_model = make_rule_oracle(env_rb, num_cursors=num_cursors)
    rb_result = benchmark_model(
        env_rb, rb_model, max_steps=3000,
        snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
    )
    print(rb_result.summary())
    Image.fromarray(env_rb.render()).save(args.out / "oracle_final.png")

    if args.oracle_only:
        return

    # --- collect BC data ---
    N = args.cols * args.rows
    K = N if num_cursors is None else num_cursors
    print(f"\n=== Collecting data ({args.rollouts} rollouts, K={K if K else N} cursors) ===")

    X, Y = collect_oracle_data(
        lambda: _make_env(args.seed),
        n_rollouts=args.rollouts,
        num_cursors=num_cursors,
    )
    print(f"Collected {len(X)} samples  feat={X.shape[1]}  act={Y.shape[1]}")

    if len(X) == 0:
        print("No samples — aborting.")
        return

    # Normalise (per-feature mean/std)
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True) + 1e-6
    X_norm = (X - mu) / sigma

    # --- train ---
    print(f"\n=== Training BC MLP (hidden={args.hidden}, epochs={args.epochs}) ===")
    mlp = _MLP(in_dim=BC_FEAT_DIM, hidden=args.hidden, out_dim=BC_ACT_DIM, seed=args.seed)
    mlp.fit(X_norm, Y, lr=1e-3, epochs=args.epochs, batch_size=256, verbose=True)

    # --- evaluate BC model ---
    print("\n=== Evaluating BC oracle ===")
    puzzle, layout = _make_env(args.seed)
    env_bc = JigsawEnvironment(puzzle, layout)
    bc_model = make_bc_oracle(
        mlp, env_bc,
        num_cursors=num_cursors,
        frame_mu=mu[0], frame_sigma=sigma[0],
    )
    bc_result = benchmark_model(
        env_bc, bc_model, max_steps=3000,
        snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
    )
    print(bc_result.summary())
    Image.fromarray(env_bc.render()).save(args.out / "bc_final.png")

    print("\n=== Summary ===")
    print(f"  Rule-based  piecewise_score={rb_result.piecewise_score:.4f}  steps={rb_result.steps}")
    print(f"  BC oracle   piecewise_score={bc_result.piecewise_score:.4f}  steps={bc_result.steps}")


if __name__ == "__main__":
    main()
