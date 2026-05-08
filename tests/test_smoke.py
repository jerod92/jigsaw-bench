"""Smoke tests — exercise the pipeline on a generated image (no dataset download)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from jigsaw_bench import (
    ActionPoint, JigsawEnvironment, generate_cuts, generate_puzzle,
    shuffle_pieces, benchmark_model,
)
from jigsaw_bench.geometry import polylines_intersect


def _gradient_image(w: int, h: int) -> Image.Image:
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    arr = np.stack([np.tile(x, (h, 1)),
                    np.tile(y[:, None], (1, w)),
                    ((np.tile(x, (h, 1)).astype(int) + np.tile(y[:, None], (1, w)).astype(int)) // 2).astype(np.uint8)], axis=-1)
    return Image.fromarray(arr)


def test_cuts_no_intersection():
    cuts = generate_cuts(width=600, height=400, n_cols=6, n_rows=4, seed=0)
    polylines = cuts.all_polylines()
    for i, (kind_a, idx_a, line_a) in enumerate(polylines):
        for j in range(i + 1, len(polylines)):
            kind_b, idx_b, line_b = polylines[j]
            shared = []
            if kind_a != kind_b:
                h_row = idx_a if kind_a == "h" else idx_b
                v_col = idx_b if kind_a == "h" else idx_a
                shared = [cuts.anchors[h_row, v_col]]
            assert not polylines_intersect(line_a, line_b, shared_endpoints=shared), \
                f"cuts ({kind_a},{idx_a}) and ({kind_b},{idx_b}) intersect outside anchors"


def test_generate_puzzle_pieces():
    img = _gradient_image(600, 400)
    puzzle = generate_puzzle(img, width=600, height=400, n_cols=6, n_rows=4, seed=0)
    assert len(puzzle.pieces) == 24
    for p in puzzle.pieces:
        assert p.sprite_rgba.shape[2] == 4
        assert p.sprite_rgba.shape[0] > 0 and p.sprite_rgba.shape[1] > 0
        assert (p.sprite_rgba[..., 3] > 0).any()


def test_shuffle_no_overlap():
    img = _gradient_image(600, 400)
    puzzle = generate_puzzle(img, width=600, height=400, n_cols=4, n_rows=3, seed=0)
    layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
    aabbs = [p.aabb_canvas for p in layout.placements]
    for i, a in enumerate(aabbs):
        for j in range(i + 1, len(aabbs)):
            b = aabbs[j]
            assert (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]), \
                f"piece {i} and {j} overlap"


def test_environment_render_and_step():
    img = _gradient_image(600, 400)
    puzzle = generate_puzzle(img, width=600, height=400, n_cols=4, n_rows=3, seed=0)
    layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
    env = JigsawEnvironment(puzzle, layout)
    frame = env.reset()
    assert frame.shape == (layout.canvas_height, layout.canvas_width, 3)
    # one cursor: hover over piece 0's shuffled centroid and grab
    p0 = layout.placements[0]
    sx, sy = p0.shuffle_centroid
    cw, ch = env.canvas_w, env.canvas_h
    obs = env.step({0: ActionPoint(sx / cw, sy / ch, grab=True)})
    assert obs.shape == (ch, cw, 3)


def test_benchmark_runs():
    img = _gradient_image(600, 400)
    puzzle = generate_puzzle(img, width=600, height=400, n_cols=4, n_rows=3, seed=0)
    layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
    env = JigsawEnvironment(puzzle, layout)

    def lazy_model(obs, step):
        return {0: ActionPoint(0.5, 0.5, grab=False)}

    result = benchmark_model(env, lazy_model, max_steps=10, snap_to=False)
    assert result.steps == 10
    assert not result.solved
    assert 0.0 <= result.piecewise_score <= 1.0
