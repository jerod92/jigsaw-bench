# Sub-agent VLM runs — May 2026

A series of small experiments using a Claude Code sub-agent as the VLM,
driving the env via `examples/vlm_driver.py`. The sub-agent reads
`/tmp/jb_frame.png` natively (Claude has vision) and emits one tool call per
step via Bash. State is persisted between calls in `/tmp/jb_state.pkl`.

## Results

| # | Image | Visibility | Rotations | Step cap | Steps used | Pieces snapped | Solved |
|---|---|---|---|---|---|---|---|
| 1 | Synthetic 4-color + center circle | Debug JSON exposed | quarter turns | 40 | 18 | 4/4 | yes |
| 2 | Synthetic 4-color + center circle | Vision-only | quarter turns | 50 | 37 | 4/4 | yes |
| 3 | Real photo (raccoon) | Vision-only | quarter turns | 60 | 56 (timed out at 77 tool calls) | 1/4 | no |
| 4 | Real photo (raccoon) | Vision-only | quarter turns | 30 | 30 | 0/4 | no |
| 5 | Real photo (raccoon) — **post-fix** | Vision-only | continuous uniform | 50 | 50 | 0/4 | no |

Run #5 is the first run with all the render/shuffle bugs fixed
(see "Caveats" below): tabs are no longer clipped, rotations are drawn from
a continuous uniform distribution, and the canvas-to-puzzle ratio is ~3:1.

## Findings

**The framework grades difficulty correctly.** The synthetic image with a
clear orientation fiducial (the central white circle + black outline, which
each piece carries a quarter-arc of) is near-trivially solvable by a Claude
sub-agent in vision-only mode. Take the fiducial away and replace the
high-contrast color quadrants with a real photograph and the same agent
struggles substantially.

**Rotation discovery is the dominant failure mode** — and it gets worse with
continuous rotations. In run 4 (quarter-turn rotations) pieces dropped within
~75 px of their actual snap targets — just inside the 80 px position threshold
— but were stuck at one of the 90/180/270 cardinal rotations. With continuous
rotations (run 5), the agent's per-piece final state shows position errors of
22–112 px (closer than run 4) but rotation errors of 58–142 deg, well outside
the 20 deg snap window. Estimating the exact tilt of a torn-edge crop of
natural imagery is genuinely hard for a multimodal LLM.

**Vision token cost is real.** Each image Read consumes vision tokens. Run 3
ate through its agent token budget at 77 tool calls. Run 5 (with explicit
"minimize Reads" guidance) used the full step budget but stayed within the
agent token cap.

**The synthetic-vs-natural-image gap is exactly the headroom you'd want from
a benchmark.** Easy mode confirms the plumbing works; hard mode leaves
plenty of room for stronger models or fine-tuned controllers to differentiate
themselves.

## Caveats — known bugs at the time of runs 1-4

Runs 1-4 were performed before two render/shuffle bugs were fixed in commit
`ab0ae11`, plus a canvas-ratio change in `f72e2da`:

1. **Tab clipping in render.** PIL's `Image.rotate(center=..., expand=True)`
   sizes the output canvas as if rotating around the image center, then
   rotates around the custom center — content can fall outside the canvas
   and get clipped. The renderer now rotates around image center and
   translates so the piece centroid lands at the desired position, leaving
   tabs intact.
2. **Discrete rotations.** Initial rotations were drawn from
   `(0, 90, 180, 270)` to keep the worst-case rotated bbox small. The default
   is now a continuous uniform distribution over `[0, 360)`. Easy mode is
   accordingly harder.
3. **Canvas:puzzle ratio.** The previous initial-canvas formula reserved
   three cells of buffer around the silhouette, blowing the canvas up to
   8-9x puzzle dims for small grids. The board now sits at ~1/3 of canvas
   for typical configs.

Run 5 is post-all-fixes.

## Artifacts

- `raccoon_initial.png` — fresh canvas at step 0 of runs 3 and 4 (pre-fix).
- `raccoon_60step_final_1of4.png` — state after run #3 timed out.
  Top-left quadrant snapped cleanly; remaining 3 pieces clustered loose
  near the center of the board.
- `raccoon_30step_final_0of4.png` — state after run #4 used its 30-step
  cap. Pieces drifted in the right direction but no snaps.
- `raccoon_postfix_initial.png` — fresh canvas at step 0 of run 5
  (post-all-fixes). Notice the prominent ~1/3-canvas board, intact tabs,
  and continuous-angle scatter.
- `raccoon_postfix_50step_final_0of4.png` — state after run #5 used its
  50-step cap. Pieces approached their slots (pos errors 22–112 px) but
  rotations missed the 20-deg snap window.

The source image used for runs #3, #4, and #5 is at
`examples/sample_images/raccoon_240x180.png` (cropped + downscaled from
`scipy.datasets.face()`).
