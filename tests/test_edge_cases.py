"""Edge-case tests for puzzle generation, shuffle, and environment.

Each test targets a specific boundary condition that could expose bugs in cut
generation, polygon rasterisation, or the shuffle layout algorithm.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from jigsaw_bench import (
    ActionPoint,
    JigsawEnvironment,
    benchmark_model,
    generate_cuts,
    generate_puzzle,
    shuffle_pieces,
)
from jigsaw_bench.geometry import polylines_intersect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(w: int, h: int, color=(128, 64, 192)) -> Image.Image:
    arr = np.full((h, w, 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


def _gradient(w: int, h: int) -> Image.Image:
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    r = np.tile(x, (h, 1))
    g = np.tile(y[:, None], (1, w))
    b = ((r.astype(int) + g.astype(int)) // 2).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


def _check_puzzle_invariants(puzzle, n_cols, n_rows):
    """Common assertions that should hold for any valid puzzle."""
    assert len(puzzle.pieces) == n_cols * n_rows
    for p in puzzle.pieces:
        assert p.sprite_rgba.shape[2] == 4, "sprite must have alpha channel"
        assert p.sprite_rgba.shape[0] > 0 and p.sprite_rgba.shape[1] > 0
        assert (p.sprite_rgba[..., 3] > 0).any(), f"piece {p.index} has empty alpha mask"
        assert p.target_centroid[0] > 0 or p.target_centroid[1] > 0  # not at canvas origin


# ---------------------------------------------------------------------------
# Cut generation edge cases
# ---------------------------------------------------------------------------

class TestCutsEdgeCases:

    def test_1x1_no_cuts(self):
        """1×1 grid produces no cuts at all."""
        cuts = generate_cuts(width=300, height=200, n_cols=1, n_rows=1, seed=0)
        assert len(cuts.h_cuts) == 0
        assert len(cuts.v_cuts) == 0
        assert cuts.all_polylines() == []

    def test_1x1_cut_no_intersection(self):
        """Empty cut-set: intersection check is vacuously satisfied."""
        cuts = generate_cuts(width=300, height=200, n_cols=1, n_rows=1, seed=0)
        polylines = cuts.all_polylines()
        # No pairs to check — just verify the list is empty
        assert len(polylines) == 0

    def test_single_row_only_v_cuts(self):
        """1-row grid: only vertical cuts, no horizontal."""
        cuts = generate_cuts(width=600, height=100, n_cols=8, n_rows=1, seed=0)
        assert len(cuts.h_cuts) == 0
        assert len(cuts.v_cuts) == 7

    def test_single_col_only_h_cuts(self):
        """1-column grid: only horizontal cuts, no vertical."""
        cuts = generate_cuts(width=100, height=600, n_cols=1, n_rows=8, seed=0)
        assert len(cuts.h_cuts) == 7
        assert len(cuts.v_cuts) == 0

    def test_single_row_v_cuts_no_intersection(self):
        """Vertical-only cuts must not intersect each other (they're parallel)."""
        cuts = generate_cuts(width=600, height=100, n_cols=8, n_rows=1, seed=0)
        polylines = cuts.all_polylines()
        for i, (_, _, la) in enumerate(polylines):
            for j in range(i + 1, len(polylines)):
                _, _, lb = polylines[j]
                assert not polylines_intersect(la, lb, shared_endpoints=[]), \
                    f"v-cuts {i} and {j} intersect in single-row grid"

    def test_single_col_h_cuts_no_intersection(self):
        """Horizontal-only cuts must not intersect each other."""
        cuts = generate_cuts(width=100, height=600, n_cols=1, n_rows=8, seed=0)
        polylines = cuts.all_polylines()
        for i, (_, _, la) in enumerate(polylines):
            for j in range(i + 1, len(polylines)):
                _, _, lb = polylines[j]
                assert not polylines_intersect(la, lb, shared_endpoints=[]), \
                    f"h-cuts {i} and {j} intersect in single-col grid"

    def test_2x2_grid(self):
        """Smallest non-trivial grid: 1 h-cut × 1 v-cut."""
        cuts = generate_cuts(width=300, height=200, n_cols=2, n_rows=2, seed=0)
        assert len(cuts.h_cuts) == 1
        assert len(cuts.v_cuts) == 1

    def test_large_grid_generates(self):
        """Large grid (30×20 = 600 cuts) completes without error."""
        cuts = generate_cuts(width=1200, height=800, n_cols=30, n_rows=20, seed=42)
        assert len(cuts.h_cuts) == 19
        assert len(cuts.v_cuts) == 29

    def test_wide_aspect_ratio(self):
        """Extreme width/height ratio (20:1) — very thin horizontal cells."""
        cuts = generate_cuts(width=1000, height=50, n_cols=20, n_rows=1, seed=0)
        assert len(cuts.v_cuts) == 19

    def test_tall_aspect_ratio(self):
        """Extreme height/width ratio — very thin vertical cells."""
        cuts = generate_cuts(width=50, height=1000, n_cols=1, n_rows=20, seed=0)
        assert len(cuts.h_cuts) == 19


# ---------------------------------------------------------------------------
# Puzzle generation edge cases
# ---------------------------------------------------------------------------

class TestPuzzleEdgeCases:

    def test_1x1_single_piece(self):
        """1×1 grid gives exactly one piece covering the whole canvas."""
        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=1, n_rows=1, seed=0)
        _check_puzzle_invariants(puzzle, 1, 1)
        p = puzzle.pieces[0]
        # The single piece should cover nearly the full canvas area
        mask_area = int((p.sprite_rgba[..., 3] > 0).sum())
        total_area = 300 * 200
        assert mask_area >= total_area * 0.95, \
            f"Single piece covers only {mask_area}/{total_area} pixels"

    def test_1x2_two_pieces(self):
        img = _gradient(400, 200)
        puzzle = generate_puzzle(img, width=400, height=200, n_cols=2, n_rows=1, seed=0)
        _check_puzzle_invariants(puzzle, 2, 1)

    def test_2x1_two_pieces(self):
        img = _gradient(200, 400)
        puzzle = generate_puzzle(img, width=200, height=400, n_cols=1, n_rows=2, seed=0)
        _check_puzzle_invariants(puzzle, 1, 2)

    def test_standard_grid(self):
        img = _gradient(600, 400)
        puzzle = generate_puzzle(img, width=600, height=400, n_cols=6, n_rows=4, seed=0)
        _check_puzzle_invariants(puzzle, 6, 4)

    def test_large_grid(self):
        """16×12 = 192 pieces on a 600×400 canvas."""
        img = _gradient(600, 400)
        puzzle = generate_puzzle(img, width=600, height=400, n_cols=16, n_rows=12, seed=7)
        _check_puzzle_invariants(puzzle, 16, 12)

    def test_prime_grid_dimensions(self):
        """7×5 grid — neither dimension divides evenly at typical canvas sizes."""
        img = _gradient(770, 550)
        puzzle = generate_puzzle(img, width=770, height=550, n_cols=7, n_rows=5, seed=42)
        _check_puzzle_invariants(puzzle, 7, 5)

    def test_small_canvas_many_pieces(self):
        """100×100 canvas with 8×8 = 64 pieces (~12 px per cell)."""
        img = _gradient(100, 100)
        puzzle = generate_puzzle(img, width=100, height=100, n_cols=8, n_rows=8, seed=3)
        _check_puzzle_invariants(puzzle, 8, 8)

    def test_tiny_cells(self):
        """200×200 canvas with 10×10 = 100 pieces (20 px per cell)."""
        img = _solid(200, 200)
        puzzle = generate_puzzle(img, width=200, height=200, n_cols=10, n_rows=10, seed=1)
        _check_puzzle_invariants(puzzle, 10, 10)

    def test_wide_puzzle(self):
        """Very wide layout: 800×60 with 20 columns × 2 rows."""
        img = _gradient(800, 60)
        puzzle = generate_puzzle(img, width=800, height=60, n_cols=20, n_rows=2, seed=5)
        _check_puzzle_invariants(puzzle, 20, 2)

    def test_tall_puzzle(self):
        """Very tall layout: 60×800 with 2 columns × 20 rows."""
        img = _gradient(60, 800)
        puzzle = generate_puzzle(img, width=60, height=800, n_cols=2, n_rows=20, seed=5)
        _check_puzzle_invariants(puzzle, 2, 20)

    def test_mask_coverage_close_to_full(self):
        """Combined piece masks should cover almost the entire canvas area."""
        img = _gradient(400, 300)
        puzzle = generate_puzzle(img, width=400, height=300, n_cols=4, n_rows=3, seed=0)
        total_alpha = sum((p.sprite_rgba[..., 3] > 0).sum() for p in puzzle.pieces)
        canvas_area = 400 * 300
        assert total_alpha >= canvas_area * 0.98, \
            f"Combined masks cover only {total_alpha}/{canvas_area} pixels"

    def test_non_overlapping_bboxes_small_grid(self):
        """Piece bounding boxes should not completely overlap on a small grid."""
        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=3, n_rows=2, seed=0)
        bboxes = [p.bbox for p in puzzle.pieces]
        # At minimum: no two identical bboxes (pieces are distinct regions)
        assert len(set(bboxes)) == len(bboxes), "Two pieces share an identical bbox"

    def test_piece_indices_contiguous(self):
        """Piece indices run 0..N-1 with no gaps."""
        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=3, n_rows=2, seed=0)
        indices = [p.index for p in puzzle.pieces]
        assert sorted(indices) == list(range(6))

    def test_reproducible_with_seed(self):
        """Same seed → same cuts → same piece bboxes."""
        img = _gradient(300, 200)
        p1 = generate_puzzle(img, width=300, height=200, n_cols=3, n_rows=2, seed=99)
        p2 = generate_puzzle(img, width=300, height=200, n_cols=3, n_rows=2, seed=99)
        for a, b in zip(p1.pieces, p2.pieces):
            assert a.bbox == b.bbox
            np.testing.assert_array_equal(a.sprite_rgba, b.sprite_rgba)

    def test_different_seeds_differ(self):
        """Different seeds → different cut shapes."""
        img = _gradient(300, 200)
        p1 = generate_puzzle(img, width=300, height=200, n_cols=3, n_rows=2, seed=0)
        p2 = generate_puzzle(img, width=300, height=200, n_cols=3, n_rows=2, seed=1)
        bboxes1 = [p.bbox for p in p1.pieces]
        bboxes2 = [p.bbox for p in p2.pieces]
        assert bboxes1 != bboxes2, "Different seeds produced identical puzzles"


# ---------------------------------------------------------------------------
# Shuffle edge cases
# ---------------------------------------------------------------------------

class TestShuffleEdgeCases:

    def test_single_piece_shuffle(self):
        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=1, n_rows=1, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        assert len(layout.placements) == 1
        p = layout.placements[0]
        # Piece should not overlap the board silhouette
        bx, by = layout.board_origin
        bw, bh = layout.board_size
        cx, cy = p.shuffle_centroid
        # The piece could be placed outside the board area (or just adjacent)
        # Just verify the layout is valid (no crash, centroid exists)
        assert np.isfinite(cx) and np.isfinite(cy)

    def test_two_piece_shuffle_no_overlap(self):
        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=2, n_rows=1, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        aabbs = [p.aabb_canvas for p in layout.placements]
        a, b = aabbs[0], aabbs[1]
        assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], \
            "Two-piece shuffle: pieces overlap"

    def test_many_pieces_no_overlap(self):
        """24-piece shuffle: no pairwise overlap."""
        img = _gradient(600, 400)
        puzzle = generate_puzzle(img, width=600, height=400, n_cols=6, n_rows=4, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        aabbs = [p.aabb_canvas for p in layout.placements]
        for i, a in enumerate(aabbs):
            for j in range(i + 1, len(aabbs)):
                b = aabbs[j]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], \
                    f"Pieces {i} and {j} overlap in shuffle"

    def test_target_centroids_inside_board(self):
        """All target centroids fall within the board (silhouette) area."""
        img = _gradient(400, 300)
        puzzle = generate_puzzle(img, width=400, height=300, n_cols=4, n_rows=3, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        bx, by = layout.board_origin
        bw, bh = layout.board_size
        for p in layout.placements:
            tx, ty = p.target_centroid
            assert bx <= tx <= bx + bw, f"target x {tx} outside board [{bx}, {bx+bw}]"
            assert by <= ty <= by + bh, f"target y {ty} outside board [{by}, {by+bh}]"

    def test_canvas_large_enough(self):
        img = _gradient(400, 300)
        puzzle = generate_puzzle(img, width=400, height=300, n_cols=4, n_rows=3, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        assert layout.canvas_width >= 400 * 2.5 * 0.9  # allow small rounding
        assert layout.canvas_height >= 300 * 2.5 * 0.9


# ---------------------------------------------------------------------------
# Environment edge cases
# ---------------------------------------------------------------------------

class TestEnvironmentEdgeCases:

    def test_single_piece_env(self):
        """Environment with 1 piece renders and steps without error."""
        img = _gradient(200, 200)
        puzzle = generate_puzzle(img, width=200, height=200, n_cols=1, n_rows=1, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        env = JigsawEnvironment(puzzle, layout)
        frame = env.reset()
        assert frame.shape[2] == 3
        obs = env.step({0: ActionPoint(0.5, 0.5, grab=False)})
        assert obs.shape == frame.shape

    def test_many_cursors_more_than_pieces(self):
        """Having more cursors than pieces should not crash."""
        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=2, n_rows=1, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        env = JigsawEnvironment(puzzle, layout)
        env.reset()
        # Send 10 cursor actions for a 2-piece puzzle
        actions = {i: ActionPoint(0.5, 0.5, grab=False) for i in range(10)}
        obs = env.step(actions)
        assert obs is not None

    def test_benchmark_single_piece(self):
        """benchmark_model works on a 1-piece puzzle."""
        img = _gradient(200, 200)
        puzzle = generate_puzzle(img, width=200, height=200, n_cols=1, n_rows=1, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        env = JigsawEnvironment(puzzle, layout)

        def lazy(obs, step):
            return {0: ActionPoint(0.5, 0.5, grab=False)}

        result = benchmark_model(env, lazy, max_steps=5, snap_to=False)
        assert result.steps == 5
        assert 0.0 <= result.piecewise_score <= 1.0

    def test_oracle_solves_small_puzzle(self):
        """Oracle should fully solve a 2×2 puzzle within budget."""
        from examples.geo_bc_oracle import make_rule_oracle

        img = _gradient(300, 200)
        puzzle = generate_puzzle(img, width=300, height=200, n_cols=2, n_rows=2, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        env = JigsawEnvironment(puzzle, layout)
        oracle = make_rule_oracle(env, num_cursors=None)   # one cursor per piece
        result = benchmark_model(
            env, oracle, max_steps=2000,
            snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
        )
        assert result.solved, f"Oracle failed to solve 2×2: score={result.piecewise_score}"

    def test_simultaneous_cursors_all_pieces(self):
        """One cursor per piece (K=N) — all move simultaneously from step 1."""
        from examples.geo_bc_oracle import make_rule_oracle

        img = _gradient(400, 300)
        puzzle = generate_puzzle(img, width=400, height=300, n_cols=4, n_rows=3, seed=0)
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=0)
        env = JigsawEnvironment(puzzle, layout)
        oracle = make_rule_oracle(env, num_cursors=None)  # K=N=12

        result = benchmark_model(
            env, oracle, max_steps=500,
            snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
        )
        # With simultaneous movement should converge faster than sequential
        assert result.piecewise_score > 0.5, \
            f"Simultaneous oracle score too low: {result.piecewise_score}"
