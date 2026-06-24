"""Generate a single puzzle from an arbitrary image and save the cut overlay + scatter."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from jigsaw_bench import generate_puzzle, shuffle_pieces, JigsawEnvironment
from jigsaw_bench.cuts import render_cuts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.cols}x{args.rows} puzzle from {args.image}")
    puzzle = generate_puzzle(
        str(args.image),
        width=args.width, height=args.height,
        n_cols=args.cols, n_rows=args.rows,
        seed=args.seed,
    )
    print(f"  pieces: {len(puzzle.pieces)}")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(puzzle.image)
    render_cuts(puzzle.cuts, ax=ax, show_anchors=False)
    fig.savefig(args.out / "cuts_over_image.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=args.seed)
    env = JigsawEnvironment(puzzle, layout)
    frame = env.render()
    Image.fromarray(frame).save(args.out / "shuffled.png")
    print(f"  outputs in {args.out}")


if __name__ == "__main__":
    main()
