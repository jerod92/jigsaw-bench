# jigsaw-bench

End-to-end jigsaw puzzle generation, scattering, and AI benchmarking.

The cut algorithm is a smoothed-spline polyline between perturbed grid anchors. The
generator validates that no two distinct cuts intersect anywhere except at the grid
anchors they share — offending cuts are regenerated until the cut-set is clean.

## Components

| Module | Purpose |
| --- | --- |
| `cuts.py` | Generate cut polylines + intersection validation |
| `puzzle.py` | Slice an input image into per-piece RGBA sprites with target metadata |
| `dataset.py` | DIV2K (train+val) downloader + batch puzzle generator |
| `shuffle.py` | Scatter pieces around a centered silhouette, no overlaps |
| `environment.py` | Multi-cursor headless interactive env (grab / rotate / translate) |
| `benchmark.py` | Score AI models against a puzzle, with optional snap-to easy mode |
| `oracle.py` | Greedy multi-cursor reference solver (any cursor/piece count) |
| `gif_utils.py` | Record a rollout and save it as an animated GIF |
| `llm_interface.py` | Cursor-style relative-motion interface for a vanilla LLM |
| `geo_model.py` | Geometric observation schema for structured (non-visual) DL models |

## Install

```bash
pip install -e .
```

## Quickstart

Generate a puzzle from any image and render the shuffled layout:

```bash
python examples/generate_single.py path/to/image.jpg --cols 12 --rows 8 --out out/
```

Run the bundled rule-based oracle:

```bash
python examples/oracle_benchmark.py path/to/image.jpg --cursors 4
```

Collect oracle demonstrations, train a behavioral cloning MLP, and evaluate it:

```bash
python examples/geo_bc_oracle.py path/to/image.jpg --rollouts 8 --epochs 300
```

## Greedy multi-cursor oracle

`GreedyOracle` is the reference solver. It handles any number of cursors `c`
(up to `MAX_CURSORS = 32`) and pieces `p` (up to `MAX_PIECES = 500`):

- **`c > p`** — every piece gets its closest cursor; spare cursors park in the corner.
- **`c == p`** — each cursor gets a unique piece.
- **`c < p`** — cursors finish a piece and pick up the next-closest one, in waves.

Each cursor aims at a guaranteed-interior point of its piece, moves **1/3 of the
remaining distance** per step (grabbing once it lands), then carries the piece —
moving 1/3 of the remaining distance **and** rotating 1/3 of the remaining
(shortest-direction) angle — until it is within snap tolerance, where it releases
and is reassigned.

```python
from jigsaw_bench import (
    generate_puzzle, shuffle_pieces, JigsawEnvironment,
    make_greedy_oracle, record_rollout, save_gif,
)

puzzle = generate_puzzle("photo.jpg", width=720, height=480, n_cols=6, n_rows=4, seed=1)
env = JigsawEnvironment(puzzle, shuffle_pieces(puzzle, canvas_scale=2.2, seed=1))

oracle = make_greedy_oracle(env, num_cursors=8)        # None → one cursor per piece
frames, result = record_rollout(env, oracle, capture_every=4, snap_to=True)
save_gif(frames, "oracle.gif", fps=14)
print(result.summary())                                 # solved=True, score=1.0
```

`examples/kaggle_oracle_demo.py` is a self-contained Kaggle script that clones
the repo and renders GIFs of the oracle solving the `c<p`, `c=p`, and `c>p`
cases plus a larger 70-piece puzzle.

## Library use

```python
from jigsaw_bench import generate_puzzle, shuffle_pieces, JigsawEnvironment, benchmark_model, ActionPoint

puzzle = generate_puzzle("photo.jpg", width=900, height=600, n_cols=12, n_rows=8, seed=0)
layout = shuffle_pieces(puzzle, canvas_scale=1.7, seed=0)
env = JigsawEnvironment(puzzle, layout)

def my_model(obs, step):
    # obs is HxWx3 uint8 — your model decides per-cursor actions
    return {0: ActionPoint(x=0.5, y=0.5, grab=False)}

result = benchmark_model(env, my_model, max_steps=2000, snap_to=True)
print(result.summary())
```

### Action format

Each cursor sends a structured action per step:

| field | type | meaning |
| --- | --- | --- |
| `x`, `y` | float in (0, 1) | absolute canvas position (normalized) |
| `grab` | bool | whether the grab button is held |
| `rotation_delta` | float in (-1, 1) | scaled to (-180°, +180°) per step |
| `render_priority` | float in (0, 1) | layering between simultaneously held pieces |
| `finished` | bool | this cursor declares done; no further inputs accepted |

Grabbed pieces always render above ungrabbed pieces. Among multiple grabbed pieces,
higher `render_priority` renders above lower.

---

## Geometric model interface

`geo_model.py` provides a structured observation schema for models that operate on
geometry rather than pixels. This is the recommended starting point for deep learning
models that are not vision-based.

### Observation schema

```python
from jigsaw_bench import geo_observation, GEO_PIECE_DIM, GEO_CURSOR_DIM

obs = geo_observation(env)  # GeoObservation
# obs.piece_features  : (N, 8) float32
# obs.cursor_features : (K, 5) float32
```

**`piece_features`** — `(N, GEO_PIECE_DIM=8)` one row per piece:

| col | feature |
| --- | --- |
| 0–1 | `cx / W`, `cy / H` — current centroid (normalised) |
| 2–3 | `sin(rot)`, `cos(rot)` — rotation as unit-circle coords |
| 4–5 | `tx / W`, `ty / H` — target (solved) centroid |
| 6–7 | `(tx−cx)/W`, `(ty−cy)/H` — positional delta to target |

**`cursor_features`** — `(K, GEO_CURSOR_DIM=5)` one row per active cursor:

| col | feature |
| --- | --- |
| 0–1 | `cursor_x / W`, `cursor_y / H` — last cursor position |
| 2 | `is_holding` — 1.0 if cursor holds a piece |
| 3–4 | centroid of held piece (normalised); 0,0 if not holding |

Supporting arrays `piece_indices` and `cursor_ids` map each row back to the
corresponding piece index / cursor id in the environment.

### Behavioral cloning oracle

`examples/geo_bc_oracle.py` trains a framework-free (NumPy-only) two-layer MLP to
clone the rule-based oracle's behavior from geometric observations.

**Per-cursor feature vector** `(BC_FEAT_DIM = 13)`:

| cols | features |
| --- | --- |
| 0–1 | cursor position normalised |
| 2 | is_holding |
| 3–4 | target piece centroid normalised |
| 5–6 | sin/cos of piece rotation |
| 7–8 | target (solved) centroid normalised |
| 9–10 | cursor → piece delta normalised |
| 11–12 | piece → target delta normalised |

**Output action vector** `(BC_ACT_DIM = 4)`:

| col | meaning |
| --- | --- |
| 0–1 | `action_x`, `action_y` — target cursor position (normalised) |
| 2 | `grab_logit` — `> 0` → `grab=True` |
| 3 | `rotation_delta` ∈ (−1, 1) |

```python
from examples.geo_bc_oracle import (
    cursor_features, _MLP, make_bc_oracle, make_rule_oracle,
)

# Train
mlp = _MLP(hidden=128)
mlp.fit(X_norm, Y, epochs=300)

# Deploy
bc_model = make_bc_oracle(mlp, env, num_cursors=4)
result = benchmark_model(env, bc_model, max_steps=3000, snap_to=True)
```

---

## Vanilla LLM mode

```python
from jigsaw_bench.llm_interface import LLMCursorInterface, benchmark_llm

# Your LLM agent: callable (frame, step, tool_schema) -> (tool_name, kwargs)
def my_agent(frame, step, tool_schema): ...

result = benchmark_llm(env, my_agent, snap_to=True)
```

The LLM operates a single cursor with relative motion (`move(dx, dy)`, `grab()`,
`release()`, `rotate(degrees)`, `finished()`). The cursor is drawn on every frame
so a multimodal model can see where it is. Easy-mode snap-to is on by default.

### Driving the env from a sub-agent / external process

For evaluations where the model lives outside Python (a separate agent, a shell loop,
an HTTP service), `examples/vlm_driver.py` is a stateless CLI that persists env state
to `/tmp/jb_state.pkl` and the latest frame to `/tmp/jb_frame.png`. By default the
printed JSON status is **vision-only** — cursor pose, grab state, held piece index,
solved flag — so the model has to actually look at the rendered frame to play. Pass
`--debug` to also include per-piece centroids/targets/errors (useful for sanity-checking
the plumbing or running an oracle).

### DIV2K batch generation

```python
from jigsaw_bench.dataset import generate_div2k_puzzles_to_disk, DIV2KConfig

generate_div2k_puzzles_to_disk(
    "out/div2k_puzzles",
    config=DIV2KConfig(splits=("train", "valid")),
    width=1200, height=900, n_cols=24, n_rows=16,
    seed=42, limit=10,
)
```

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The smoke tests assert the intersection invariant directly: every generated cut-set
is checked pairwise and no spurious crossings are allowed.
