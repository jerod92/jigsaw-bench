"""Benchmark a trivial 'oracle' model that just translates each piece to its target.

Demonstrates the multi-cursor benchmark API end to end.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from jigsaw_bench import (
    ActionPoint, JigsawEnvironment, benchmark_model, generate_puzzle, shuffle_pieces,
)


def make_oracle(env: JigsawEnvironment, num_cursors: int = 4):
    """Each cursor walks through the piece queue: grab → align → release → next."""
    targets = env.target_centroids()
    queue = list(targets.keys())
    state = {cid: {"piece": None, "phase": "pick"} for cid in range(num_cursors)}

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
                actions[cid] = ActionPoint(cx / cw, cy / ch, grab=True, render_priority=cid / max(1, num_cursors - 1))
                st["phase"] = "carry"
            elif st["phase"] == "carry":
                # Move toward target with capped step + rotate to zero.
                dx, dy = tx - cx, ty - cy
                step_len = float(np.hypot(dx, dy))
                if step_len > 1:
                    factor = min(40.0, step_len) / step_len
                    nx, ny = cx + dx * factor, cy + dy * factor
                else:
                    nx, ny = tx, ty
                rot_correction = -rot_err
                rot_norm = float(np.clip(rot_correction / 180.0, -1.0, 1.0)) * 0.2
                actions[cid] = ActionPoint(
                    nx / cw, ny / ch, grab=True,
                    rotation_delta=rot_norm,
                    render_priority=cid / max(1, num_cursors - 1),
                )
                if step_len < 0.5 and abs(rot_err) < 0.5:
                    st["phase"] = "release"
            elif st["phase"] == "release":
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                st["phase"] = "pick"
        return actions

    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--cursors", type=int, default=4)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    puzzle = generate_puzzle(str(args.image), width=900, height=600,
                             n_cols=args.cols, n_rows=args.rows, seed=0)
    layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
    env = JigsawEnvironment(puzzle, layout)
    model = make_oracle(env, num_cursors=args.cursors)

    result = benchmark_model(
        env, model,
        max_steps=2000,
        snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
        record_history=False,
    )
    print(result.summary())
    Image.fromarray(env.render()).save(args.out / "final.png")


if __name__ == "__main__":
    main()
