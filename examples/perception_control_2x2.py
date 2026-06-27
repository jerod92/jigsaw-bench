#!/usr/bin/env python3
"""A small image-conditioned model that solves 2x2 puzzles by behavioral cloning.

This is the configuration that actually *solves* on a CPU budget (see
``MULTICURSOR_BC_FINDINGS.md`` for the full investigation and why the fully
autonomous variants do not).  It uses a **perception + control** split:

    perception (LEARNED) : a CNN reads the rendered frame + the held piece's
                           cursor state and predicts that piece's HOME (tx, ty).
    control    (fixed)   : a deterministic controller moves the piece 1/3 of the
                           way toward the predicted home and rotates 1/3 toward
                           upright each step — exact, so errors don't compound.

Why this is well posed: the cursor's ``held_centroid`` feature only says where a
piece *is now* (random scatter), not which quadrant it belongs to.  The model
must therefore read the **image** to localise the home.  Predicting a fixed
target (rather than a moving waypoint) means the controller converges exactly to
whatever the model predicts — no compounding drift.

The greedy oracle owns assignment / approach / grab / release (pixel-precise
alignment a 64-96px CPU model cannot perceive); the model owns the carry motion.

Usage
-----
    python examples/perception_control_2x2.py --epochs 80 --out out/pc2x2

Requires torch (CPU is fine; a full run is a few minutes).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from jigsaw_bench import (
    ActionPoint,
    JigsawEnvironment,
    generate_puzzle,
    make_greedy_oracle,
    perimeter_cursor_starts,
    record_rollout,
    save_gif,
    shuffle_pieces,
)

FRAME = 80          # CNN input resolution
CURSORS = 4         # one per piece (2x2)
PIECE_PX = 130      # source pixels per piece
SNAP_POS = 14.0     # snap-to assist tolerance (easy mode; snaps to exact pose)
SNAP_ROT = 8.0
MOVE = 1.0 / 3.0    # controller: fraction of remaining distance/angle per step


def make_synthetic_image(path: Path) -> None:
    """A gradient with four strongly-tinted quadrants so pieces are identifiable."""
    w = h = 600
    yy, xx = np.mgrid[0:h, 0:w]
    im = np.stack(
        [255 * xx / w, 255 * yy / h, 255 * (1.0 - (xx + yy) / (w + h))], axis=2
    ).clip(0, 255).astype(np.uint8)
    tints = [(0, 0, (220, 40, 40)), (0, 1, (40, 220, 40)),
             (1, 0, (40, 40, 220)), (1, 1, (220, 220, 40))]
    for qy, qx, col in tints:
        ys = slice(qy * h // 2, (qy + 1) * h // 2)
        xs = slice(qx * w // 2, (qx + 1) * w // 2)
        im[ys, xs] = (im[ys, xs] * 0.3 + np.array(col) * 0.7).astype(np.uint8)
    Image.fromarray(im).save(path)


def make_env(img_path: Path, seed: int) -> JigsawEnvironment:
    puzzle = generate_puzzle(str(img_path), width=2 * PIECE_PX, height=2 * PIECE_PX,
                             n_cols=2, n_rows=2, seed=seed)
    layout = shuffle_pieces(puzzle, canvas_scale=2.2, seed=seed)
    starts = perimeter_cursor_starts(layout.canvas_width, layout.canvas_height, CURSORS)
    return JigsawEnvironment(puzzle, layout, initial_cursor_positions=starts)


def resize(frame: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((FRAME, FRAME), Image.BILINEAR))


def cursor_features(env: JigsawEnvironment, cid: int) -> np.ndarray:
    cw, ch = env.canvas_w, env.canvas_h
    cs = env.cursors.get(cid)
    if cs is None:
        return np.zeros(5, dtype=np.float32)
    hx, hy, hold = 0.0, 0.0, 0.0
    if cs.held_piece is not None:
        c = env.piece_centroids().get(cs.held_piece)
        if c is not None:
            hx, hy, hold = c[0] / cw, c[1] / ch, 1.0
    return np.array([cs.last_x / cw, cs.last_y / ch, hold, hx, hy], dtype=np.float32)


class HomeNet(nn.Module):
    """frame + per-cursor state -> predicted home (tx, ty) for the held piece."""

    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(2),
        )
        self.frame_proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4, 128), nn.ReLU(True))
        self.cursor_mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(True), nn.Linear(32, 32), nn.ReLU(True))
        self.head = nn.Sequential(nn.Linear(160, 128), nn.ReLU(True), nn.Linear(128, 2))

    def forward(self, frame: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        k = cur.shape[1]
        fe = self.frame_proj(self.cnn(frame)).unsqueeze(1).expand(-1, k, -1)
        return self.head(torch.cat([fe, self.cursor_mlp(cur)], dim=-1))


def collect(img_path: Path, seeds, max_steps: int = 200):
    """Record (frame, cursor_state, home_label, holding_mask) over oracle rollouts."""
    frames, curs, homes, masks = [], [], [], []
    for s in seeds:
        try:
            env = make_env(img_path, s)
        except RuntimeError:
            continue  # rare 2x2 cut-generation failure on some seeds
        env.reset()
        orc = make_greedy_oracle(env, num_cursors=CURSORS,
                                 snap_pos_tol_px=SNAP_POS, snap_rot_tol_deg=SNAP_ROT)
        tgt = env.target_centroids()
        cw, ch = env.canvas_w, env.canvas_h
        last_held: dict[int, int | None] = {}
        for step in range(1, max_steps + 1):
            raw = env.render()
            acts = orc(raw, step)
            cf = np.stack([cursor_features(env, c) for c in range(CURSORS)])
            home = np.zeros((CURSORS, 2), dtype=np.float32)
            mask = np.zeros(CURSORS, dtype=np.float32)
            for c in range(CURSORS):
                cs = env.cursors.get(c)
                if cs is not None and cs.held_piece is not None:
                    tx, ty = tgt[cs.held_piece]
                    home[c] = [tx / cw, ty / ch]
                    mask[c] = 1.0
            if mask.any():
                frames.append(resize(raw))
                curs.append(cf)
                homes.append(home)
                masks.append(mask)
            env.step(acts)
            now = {cid: c.held_piece for cid, c in env.cursors.items()}
            for cid, prev in list(last_held.items()):
                if prev is not None and prev != now.get(cid):
                    env.snap_piece(prev, SNAP_POS, SNAP_ROT)
            last_held = now
            if env.is_solved():
                break
    return (np.stack(frames).astype(np.uint8), np.stack(curs).astype(np.float32),
            np.stack(homes).astype(np.float32), np.stack(masks).astype(np.float32))


class HomeDataset(Dataset):
    def __init__(self, frames, curs, homes, masks) -> None:
        self.frames = torch.from_numpy(frames)
        self.curs = torch.from_numpy(curs)
        self.homes = torch.from_numpy(homes)
        self.masks = torch.from_numpy(masks)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i):
        return (self.frames[i].permute(2, 0, 1).float() / 255.0,
                self.curs[i], self.homes[i], self.masks[i])


def _rot_err(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


def hybrid_policy(model: HomeNet, env: JigsawEnvironment):
    """Oracle owns approach/grab/release; the model predicts home, controller carries."""
    orc = make_greedy_oracle(env, num_cursors=CURSORS,
                             snap_pos_tol_px=SNAP_POS, snap_rot_tol_deg=SNAP_ROT)
    cw, ch = env.canvas_w, env.canvas_h
    model.eval()

    def policy(obs, step):
        base = orc(obs, step)
        ft = torch.from_numpy(resize(obs).transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
        ct = torch.from_numpy(np.stack([cursor_features(env, c) for c in range(CURSORS)])).unsqueeze(0)
        with torch.no_grad():
            home = model(ft, ct)[0].numpy()
        rots = env.piece_rotations()
        for c in range(CURSORS):
            cs = env.cursors.get(c)
            ap = base.get(c)
            if cs is not None and cs.held_piece is not None and ap is not None and ap.grab:
                p = cs.held_piece
                tx, ty = float(home[c, 0]) * cw, float(home[c, 1]) * ch
                offx, offy = cs.grab_offset_local if cs.grab_offset_local else (0.0, 0.0)
                cursor_tx, cursor_ty = tx + offx, ty + offy   # cursor goal so piece lands home
                nx = cs.last_x + (cursor_tx - cs.last_x) * MOVE
                ny = cs.last_y + (cursor_ty - cs.last_y) * MOVE
                rstep = (-_rot_err(rots[p])) * MOVE
                base[c] = ActionPoint(
                    float(np.clip(nx / cw, 0, 1)), float(np.clip(ny / ch, 0, 1)), grab=True,
                    rotation_delta=float(np.clip(rstep / env.max_step_rotation_deg, -1, 1)),
                    render_priority=c / 3.0,
                )
        return base

    return policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("out/pc2x2"))
    ap.add_argument("--train-seeds", type=int, default=220)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    img_path = args.out / "source.png"
    if not img_path.exists():
        make_synthetic_image(img_path)

    t0 = time.time()
    tr = collect(img_path, range(args.train_seeds))
    va = collect(img_path, range(500, 512))
    print(f"home samples: train={len(tr[0])} val={len(va[0])}  ({time.time() - t0:.0f}s)")

    tl = DataLoader(HomeDataset(*tr), batch_size=64, shuffle=True)
    vl = DataLoader(HomeDataset(*va), batch_size=64)
    model = HomeNet()
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def masked_mse(pred, target, mask):
        return (((pred - target) ** 2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)

    best, best_sd = float("inf"), None
    for ep in range(1, args.epochs + 1):
        model.train()
        for f, c, t, m in tl:
            loss = masked_mse(model(f, c), t, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        va_loss, n = 0.0, 0
        with torch.no_grad():
            for f, c, t, m in vl:
                va_loss += masked_mse(model(f, c), t, m).item() * len(f)
                n += len(f)
        va_loss /= max(1, n)
        if va_loss < best:
            best = va_loss
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}/{args.epochs}  val={va_loss:.5f}")
    if best_sd is not None:
        model.load_state_dict(best_sd)
    torch.save(model.state_dict(), args.out / "home.pt")

    print("\n=== Evaluating perception+control policy ===")
    solved, tried, scores = 0, 0, []
    for seed in [0, 1, 2, 3, 4, 500, 501, 502]:
        try:
            env = make_env(img_path, seed)
        except RuntimeError:
            continue
        tried += 1
        frames, res = record_rollout(env, hybrid_policy(model, env), max_steps=160, capture_every=2,
                                     snap_to=True, snap_pos_threshold_px=SNAP_POS,
                                     snap_rot_threshold_deg=SNAP_ROT)
        solved += int(res.solved)
        scores.append(res.piecewise_score)
        print(f"  seed {seed}: solved={res.solved} steps={res.steps} score={res.piecewise_score:.3f}")
        save_gif(frames, args.out / f"pc_{seed}.gif", fps=10, max_width=420)
    print(f"\nSOLVED {solved}/{tried}  mean_score={np.mean(scores):.3f}")
    print(f"Outputs in {args.out}")


if __name__ == "__main__":
    main()
