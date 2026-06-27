# Multi-cursor behavioral cloning: what learns, and what doesn't

This note records an investigation into training an image-conditioned policy to
solve jigsaw puzzles by **behavioral cloning** (BC) of the `GreedyOracle`. The
goal was a model that drives the multi-cursor environment from the **rendered
frame** (plus per-cursor state), never from raw piece coordinates.

All runs are on a 2x2 puzzle (4 pieces, 4 cursors), trained on CPU. The model is
one network evaluated once per step: a shared CNN frame embedding broadcast to
K cursor heads.

## TL;DR

- The **continuous, perceptual sub-skills are learnable** and generalize:
  carrying a held piece to its home, derotating it, and (with the cursor drawn
  on the frame) deciding when to grab/release.
- The one piece that does **not** yield to BC at this data/compute scale is the
  **multi-object approach assignment** — "which scattered piece is *mine*?" It is
  a generalization-from-limited-data problem entangled with assignment ambiguity,
  and it is robust to the output parameterization (we tried three).
- The configuration that **solves on a CPU budget** uses a perception + control
  split (`examples/perception_control_2x2.py`): the model predicts each held
  piece's home from the image; a fixed controller carries it there. It solves
  **4/8** held-out puzzles with **mean piecewise score 0.846**.

## Sub-skill diagnosis (one autonomous network, multiple heads)

Trained end-to-end with per-head losses; reported on held-out seeds:

| Sub-skill | Generalizes? | Evidence |
| --- | --- | --- |
| Carry (move held piece toward home) | yes | val MSE ~6e-4 |
| Rotation / derotate | yes | val MSE ~3e-3 |
| Home perception ("where does this piece belong") | yes | aux head val ~9e-4 |
| Grab / release timing | yes — **once the cursor is drawn on the frame** | grab logit **+2.9 over a piece vs −8.6 off** |
| Approach: localize *my* target piece | **no** | model emits an averaged target; cursor never reaches a piece |

Key enabler for grab: the environment render does not show the cursor, so
"am I on a piece?" is not perceivable from `(frame, cursor xy)` alone. Drawing a
marker at each cursor turns it into a local figure-ground question, and the grab
head then learns it cleanly.

## The approach wall, and three output parameterizations

The approach target is multimodal (several plausible pieces) and the assignment
is hidden, so the model regresses toward the mean. We tried, in order:

1. **Continuous regression (waypoint).** Cursor drifts to the board center; gets
   within ~29px of a piece then wanders off (no stable fixed point). 0/10.
2. **Continuous regression (absolute destination).** Removes compounding drift
   (env action space is absolute positioning, so re-targeting a fixed point is
   self-correcting), but the predicted destination is an average → cursor heads
   to the middle. 0/10.
3. **Factorized multi-discrete** (a softmax over bins for x and a separate one for
   y, per cursor; argmax). Helps: ~70% of cursors now aim within ~60px of their
   piece. But a per-axis product `P(x)·P(y)` invents **ghost modes**: under
   assignment uncertainty between pieces A `(xA,yA)` and B `(xB,yB)`, the four
   product peaks include the non-existent `(xA,yB)` and `(xB,yA)`; argmax
   sometimes lands on a ghost (the ~600px aim errors). 0/10.
4. **Joint 2D heatmap** (one softmax over the whole `NBINS×NBINS` grid; no cross
   terms, so no ghosts in principle) + a within-cell residual for sub-bin
   precision. Theoretically the right fix, but the `NBINS²`-way softmax is far
   more data-hungry: with ~7k samples from 260 seeds it overfits from epoch 1
   and the argmax is diffuse, so in this regime it underperforms the factorized
   head. 0/10.

The consistent conclusion: the lever is **data scale**, not the output head.
Grab, carry, rotation, and home perception all generalize; approach assignment
does not, from a few hundred seeds on CPU.

## What solves: perception + control

`examples/perception_control_2x2.py`. The greedy oracle owns
assignment/approach/grab/release (pixel-precise alignment a small CPU model
can't perceive); the **model owns the visually-grounded carry**: it reads the
frame, predicts the held piece's home, and a deterministic controller drives the
piece there. Predicting a fixed target (not a moving waypoint) means no
compounding error. Result on held-out seeds: **4/8 solved, mean score 0.846**;
the misses are 1–2 mislocalized pieces (~41px mean home error), which closes
with more resolution/data.

## If pushing further (not done here)

- **Scale the data** (thousands of seeds) on GPU — the 2D heatmap likely *does*
  win once it can become peaked; this is the clean test of the ghost-mode fix.
- **Reinforcement learning** with the env's dense `piecewise_score` reward, so
  the policy learns assignment from outcome instead of cloning a hidden one.
- **Slot-attention / iterative assignment** architecture that explicitly binds
  cursors to pieces rather than regressing a target.

## Reproduce

```bash
# the solving model (a few minutes on CPU)
python examples/perception_control_2x2.py --epochs 80 --out out/pc2x2

# the reference solver this clones, for comparison
python examples/kaggle_oracle_demo.py   # GreedyOracle, solves to score 1.0
```
