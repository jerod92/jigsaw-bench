"""Tests for the greedy multi-cursor oracle."""
import numpy as np
import pytest
from PIL import Image

from jigsaw_bench import (
    JigsawEnvironment,
    MAX_CURSORS,
    benchmark_model,
    generate_puzzle,
    make_greedy_oracle,
    perimeter_cursor_starts,
    shuffle_pieces,
)


@pytest.fixture(scope="module")
def image_path(tmp_path_factory):
    w, h = 480, 320
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [255 * xx / w, 255 * yy / h, 255 * (1.0 - (xx + yy) / (w + h))], axis=2
    ).clip(0, 255).astype(np.uint8)
    path = tmp_path_factory.mktemp("img") / "src.png"
    Image.fromarray(img).save(path)
    return str(path)


def _make_env(image_path, n_cols=4, n_rows=3, seed=1):
    puzzle = generate_puzzle(image_path, width=n_cols * 90, height=n_rows * 90,
                             n_cols=n_cols, n_rows=n_rows, seed=seed)
    layout = shuffle_pieces(puzzle, canvas_scale=2.2, seed=seed)
    return JigsawEnvironment(puzzle, layout)


def _run(env, cursors):
    oracle = make_greedy_oracle(env, num_cursors=cursors,
                                snap_pos_tol_px=14, snap_rot_tol_deg=8)
    return oracle, benchmark_model(
        env, oracle, max_steps=3000,
        snap_to=True, snap_pos_threshold_px=14, snap_rot_threshold_deg=8,
    )


@pytest.mark.parametrize("cursors", [4, 12, 20])
def test_solves_all_assignment_cases(image_path, cursors):
    """c<p, c=p and c>p (12 pieces) all reach a fully-solved board."""
    env = _make_env(image_path)  # 4*3 = 12 pieces
    _, result = _run(env, cursors)
    assert result.solved
    assert result.piecewise_score == pytest.approx(1.0)


def test_single_cursor_solves_sequentially(image_path):
    env = _make_env(image_path)
    oracle, result = _run(env, 1)
    assert oracle.K == 1
    assert result.solved


def test_cursor_count_capped_at_max(image_path):
    env = _make_env(image_path, n_cols=8, n_rows=8)  # 64 pieces
    oracle = make_greedy_oracle(env, num_cursors=999)
    assert oracle.K == MAX_CURSORS


def test_default_is_one_cursor_per_piece(image_path):
    env = _make_env(image_path)  # 12 pieces, < MAX_CURSORS
    oracle = make_greedy_oracle(env)
    assert oracle.K == 12


def test_distinct_cursor_starts(image_path):
    """Spread-out starts make cursors distinguishable from step 1, oracle still solves."""
    env = _make_env(image_path)  # 12 pieces
    starts = perimeter_cursor_starts(env.canvas_w, env.canvas_h, 12)
    env.initial_cursor_positions = starts
    env.reset()
    positions = {(round(env.cursors[c].last_x), round(env.cursors[c].last_y)) for c in range(12)}
    assert len(positions) == 12  # all distinct
    assert (0, 0) not in positions
    _, result = _run(env, 12)
    assert result.solved


def test_perimeter_starts_within_canvas(image_path):
    starts = perimeter_cursor_starts(800, 600, 10)
    assert len(starts) == 10
    for x, y in starts.values():
        assert 0 < x < 800 and 0 < y < 600


def test_held_piece_cannot_be_stolen(image_path):
    """Two cursors aiming at the same point don't both end up holding one piece."""
    env = _make_env(image_path)
    env.reset()
    from jigsaw_bench import ActionPoint
    centroids = env.piece_centroids()
    idx = sorted(centroids)[0]
    cx, cy = centroids[idx]
    # Cursor 0 grabs the piece; cursor 1 then aims at the same spot.
    env.step({0: ActionPoint(cx / env.canvas_w, cy / env.canvas_h, grab=True)})
    env.step({
        0: ActionPoint(cx / env.canvas_w, cy / env.canvas_h, grab=True),
        1: ActionPoint(cx / env.canvas_w, cy / env.canvas_h, grab=True),
    })
    assert env.cursors[0].held_piece == idx
    assert env.cursors[1].held_piece != idx  # could be None or a different piece
