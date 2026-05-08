# Sub-agent VLM runs — May 2026

A series of small experiments using a Claude Code sub-agent as the VLM,
driving the env via `examples/vlm_driver.py`. The sub-agent reads
`/tmp/jb_frame.png` natively (Claude has vision) and emits one tool call per
step via Bash. State is persisted between calls in `/tmp/jb_state.pkl`.

All four runs use the same 2x2 puzzle config: 240x180 source image,
canvas_scale=2.0, quarter-turn rotations only, snap thresholds 80 px / 20 deg.

## Results

| # | Image | Visibility | Step cap | Steps used | Pieces snapped | Solved |
|---|---|---|---|---|---|---|
| 1 | Synthetic 4-color + center circle | Debug JSON exposed | 40 | 18 | 4/4 | yes |
| 2 | Synthetic 4-color + center circle | Vision-only | 50 | 37 | 4/4 | yes |
| 3 | Real photo (raccoon) | Vision-only | 60 | 56 (timed out at 77 tool calls) | 1/4 | no |
| 4 | Real photo (raccoon) | Vision-only | 30 | 30 | 0/4 | no |

## Findings

**The framework graded the difficulty correctly.** The synthetic image with a
clear orientation fiducial (the central white circle + black outline, which
each piece carries a quarter-arc of) is near-trivially solvable by a Claude
sub-agent in vision-only mode. Take the fiducial away and replace the high-
contrast color quadrants with a real photograph and the same agent struggles.

**The dominant failure mode is rotation discovery, not position.** In run 4,
the agent dropped pieces within ~75 px of their actual snap targets — just
inside the 80 px position threshold — but pieces were still rotated 90/180/270
deg from upright. Without a fiducial, the agent has no efficient way to read
piece orientation from a torn-edge crop of natural imagery, so it falls back
to expensive guess-and-check rotation cycles that exhaust the step budget.

**Vision token cost is real.** Each image Read consumes vision tokens. The
60-step run (#3) was instructed normally and ate through its agent token
budget at 77 tool calls. The 30-step run (#4) was given an explicit "minimize
Reads" directive and stayed within budget but ran out of steps before
discovering rotations.

**The synthetic vs. natural-image gap is exactly the headroom you'd want
from a benchmark.** Easy mode confirms the plumbing works; hard mode leaves
plenty of room for stronger models or fine-tuned controllers to differentiate
themselves.

## Artifacts

- `raccoon_initial.png` — fresh canvas at step 0 of the raccoon runs.
- `raccoon_60step_final_1of4.png` — state after run #3 timed out.
  Top-left quadrant snapped cleanly; remaining 3 pieces clustered loose
  near the center of the board.
- `raccoon_30step_final_0of4.png` — state after run #4 used its 30-step
  cap. Pieces drifted in the right direction but no snaps.

The source image used for runs #3 and #4 is at
`examples/sample_images/raccoon_240x180.png` (cropped + downscaled from
`scipy.datasets.face()`).
