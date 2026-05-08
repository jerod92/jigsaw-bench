# jigsaw-bench

End-to-end jigsaw puzzle generation, scattering, and AI benchmarking.

The cut algorithm is a smoothed-spline polyline between perturbed grid anchors. The
generator now validates that no two distinct cuts intersect anywhere except at the
grid anchors they share — offending cuts are regenerated until the cut-set is clean.

## Components

| Module | Purpose |
| --- | --- |
| `cuts.py` | Generate cut polylines + intersection validation |
| `puzzle.py` | Slice an input image into per-piece RGBA sprites with target metadata |
| `dataset.py` | DIV2K (train+val) downloader + batch puzzle generator |
| `shuffle.py` | Scatter pieces around a centered silhouette, no overlaps |
| `environment.py` | Multi-cursor headless interactive env (grab / rotate / translate) |
| `benchmark.py` | Score AI models against a puzzle, with optional snap-to easy mode |
| `llm_interface.py` | Cursor-style relative-motion interface for a vanilla LLM |

## Install

```bash
pip install -e .
```

## Quickstart

Generate a puzzle from any image and render the shuffled layout:

```bash
python examples/generate_single.py path/to/image.jpg --cols 12 --rows 8 --out out/
```

Run the bundled "oracle" benchmark (translates each piece to its target):

```bash
python examples/oracle_benchmark.py path/to/image.jpg --cursors 4
```

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

### Vanilla LLM mode

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

For evaluations where the model lives outside Python (a separate Claude Code agent,
a shell loop, an HTTP service), `examples/vlm_driver.py` is a stateless CLI that
persists env state to `/tmp/jb_state.pkl` and the latest frame to `/tmp/jb_frame.png`.
By default the printed JSON status is **vision-only** — cursor pose, grab state, held
piece index, solved flag — so the model has to actually look at the rendered frame
to play. Pass `--debug` to also include per-piece centroids/targets/errors (useful
for sanity-checking the plumbing or running an oracle).

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
