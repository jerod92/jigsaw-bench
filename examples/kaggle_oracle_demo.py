#!/usr/bin/env python3
"""Kaggle oracle demo: watch the greedy multi-cursor oracle solve puzzles.

Clones the repo, builds a few puzzles, and runs :class:`GreedyOracle` for a
range of cursor counts — saving an animated GIF of each rollout so you can
watch the assignment / approach / grab / carry / release logic in action.

No training, no GPU required — this is purely an oracle visualisation.

Requirements
------------
- Internet access: ON (for the git clone)

Run from a Kaggle notebook cell::

    !python /kaggle/working/jigsaw-bench/examples/kaggle_oracle_demo.py

Outputs (in /kaggle/working/jigsaw_oracle/):
    oracle_c<C>_p<P>.gif   — animated rollout per config
    oracle_c<C>_p<P>.png   — final solved frame per config
    results.json           — per-config solved / steps / score
"""
from __future__ import annotations

# ── Setup: clone + install ───────────────────────────────────────────────────
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/jigsaw-bench")

if not REPO_DIR.exists():
    print("Cloning jigsaw-bench …")
    subprocess.run(
        ["git", "clone", "https://github.com/jerod92/jigsaw-bench.git", str(REPO_DIR)],
        check=True,
    )
    print("Installing package …")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "-q"],
        check=True,
    )
else:
    print(f"Repo already present at {REPO_DIR}")

sys.path.insert(0, str(REPO_DIR))

# ── Imports ──────────────────────────────────────────────────────────────────
import json
import time

import numpy as np
from PIL import Image

from jigsaw_bench import (
    JigsawEnvironment,
    MAX_CURSORS,
    MAX_PIECES,
    generate_puzzle,
    make_greedy_oracle,
    record_rollout,
    save_gif,
    shuffle_pieces,
)

OUT_DIR = Path("/kaggle/working/jigsaw_oracle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Snap tolerances — the oracle releases within these, the benchmark snaps exact.
SNAP_POS_PX = 14.0
SNAP_ROT_DEG = 8.0

# Per-piece pixel budget so pieces stay crisp as the grid grows (image fidelity).
PIECE_PX = 120

print(f"MAX_CURSORS={MAX_CURSORS}  MAX_PIECES={MAX_PIECES}")
print(f"Output: {OUT_DIR}")

# ── Synthetic source image ───────────────────────────────────────────────────
IMG_PATH = OUT_DIR / "source.png"


def make_synthetic_image(path: Path, width: int = 1600, height: int = 1100) -> None:
    """A smooth gradient + random blocks — crisp at any piece count, no asset needed."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack(
        [255 * xx / width, 255 * yy / height, 255 * (1.0 - (xx + yy) / (width + height))],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    for _ in range(80):
        x1, y1 = int(rng.integers(0, width - 100)), int(rng.integers(0, height - 100))
        x2 = min(x1 + int(rng.integers(60, 320)), width)
        y2 = min(y1 + int(rng.integers(60, 220)), height)
        color = rng.integers(40, 255, size=3).astype(np.float32)
        alpha = float(rng.uniform(0.2, 0.6))
        patch = img[y1:y2, x1:x2].astype(np.float32)
        img[y1:y2, x1:x2] = np.clip(patch * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"Synthetic image: {path}  ({width}×{height})")


if not IMG_PATH.exists():
    make_synthetic_image(IMG_PATH)


# ── One oracle rollout → GIF ─────────────────────────────────────────────────

def run_config(n_cols: int, n_rows: int, cursors: int, *, seed: int = 1) -> dict:
    """Build a puzzle, run the greedy oracle, save a GIF + final PNG."""
    n_pieces = n_cols * n_rows
    pw, ph = n_cols * PIECE_PX, n_rows * PIECE_PX

    puzzle = generate_puzzle(str(IMG_PATH), width=pw, height=ph,
                             n_cols=n_cols, n_rows=n_rows, seed=seed)
    layout = shuffle_pieces(puzzle, canvas_scale=2.2, seed=seed)
    env = JigsawEnvironment(puzzle, layout)

    oracle = make_greedy_oracle(
        env, num_cursors=cursors,
        snap_pos_tol_px=SNAP_POS_PX, snap_rot_tol_deg=SNAP_ROT_DEG,
    )
    eff_c = oracle.K
    case = "c<p" if eff_c < n_pieces else ("c=p" if eff_c == n_pieces else "c>p")

    print(f"\n=== c={cursors} (effective {eff_c}), p={n_pieces}  [{case}] ===")
    t0 = time.time()
    frames, result = record_rollout(
        env, oracle, max_steps=4000, capture_every=4,
        snap_to=True, snap_pos_threshold_px=SNAP_POS_PX, snap_rot_threshold_deg=SNAP_ROT_DEG,
    )
    summary = result.summary()
    print(f"  solved={summary['solved']}  steps={summary['steps']}  "
          f"score={summary['piecewise_score']:.4f}  ({time.time() - t0:.1f}s)")

    tag = f"c{eff_c}_p{n_pieces}"
    save_gif(frames, OUT_DIR / f"oracle_{tag}.gif", fps=14, max_width=720)
    Image.fromarray(frames[-1]).save(OUT_DIR / f"oracle_{tag}.png")

    return {"requested_cursors": cursors, "effective_cursors": eff_c,
            "pieces": n_pieces, "case": case, **summary}


def main() -> None:
    # Demonstrate all three assignment branches on a 24-piece puzzle, plus a
    # larger 70-piece puzzle to show reassignment waves and image fidelity.
    configs = [
        (6, 4, 8),    # c < p  — cursors finish and pick up the next-closest piece
        (6, 4, 24),   # c = p  — every cursor gets a unique piece
        (6, 4, 32),   # c > p  — 8 cursors are disemployed to the corner
        (10, 7, 32),  # c < p  — 70 pieces, several reassignment waves
    ]

    results = [run_config(c, r, k) for (c, r, k) in configs]

    print("\n┌──────────────────────────────────────────────────────────────┐")
    print("│  cursors(eff)  pieces   case   solved   steps    score        │")
    print("├──────────────────────────────────────────────────────────────┤")
    for r in results:
        print(f"│   {r['effective_cursors']:3d}          {r['pieces']:4d}    "
              f"{r['case']:4s}    {str(r['solved']):5s}   {r['steps']:5d}   "
              f"{r['piecewise_score']:.4f}       │")
    print("└──────────────────────────────────────────────────────────────┘")

    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nGIFs + results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
