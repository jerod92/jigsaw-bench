#!/usr/bin/env python3
"""Kaggle training: behavioral cloning of the greedy multi-cursor oracle.

Clones the repo, collects demonstrations from :class:`GreedyOracle`, trains a
PyTorch model that drives **all K cursors from one shared visual embedding**,
then benchmarks the learned policy against the oracle and saves GIFs of both.

Requirements
------------
- Kaggle accelerator: GPU (T4/P100) strongly recommended — CPU works but is slow.
- Internet access: ON (for the git clone).

Run::

    !python /kaggle/working/jigsaw-bench/examples/kaggle_multicursor_train.py

Design
------
The model is a single network evaluated **once per step**::

    frame (B, 3, H, W)  → CNN → 512-d shared embedding ─┐
                                                          ├→ (B, K, 4) actions
    cursors (B, K, 5)   → MLP → (B, K, 64) per-cursor  ─┘

It sees only the rendered frame and per-cursor state ``[x/W, y/H, is_holding,
held_cx/W, held_cy/H]`` — never raw piece coordinates.  Cursors **start at
distinct spread-out positions** (``perimeter_cursor_starts``), so "move toward
your nearest piece, grab it, carry it home, release" is a well-posed function of
what the model can actually see.  The learned policy does everything — there is
no piece-assignment state machine at inference.

Per-cursor action (``ACT_DIM = 4``):
    [0] action_x   [1] action_y   [2] grab_logit (>0 → grab)   [3] rotation_delta
"""
from __future__ import annotations

# ── Setup: clone + install ───────────────────────────────────────────────────
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/working/jigsaw-bench")

if not REPO_DIR.exists():
    print("Cloning jigsaw-bench …")
    subprocess.run(
        ["git", "clone", "https://github.com/jerod92/jigsaw-bench.git", str(REPO_DIR)],
        check=True,
    )
    print("Installing package …")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_DIR), "-q"],
        check=True,
    )
else:
    print(f"Repo already present at {REPO_DIR}")

sys.path.insert(0, str(REPO_DIR))

# ── Imports ──────────────────────────────────────────────────────────────────
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# ── Configuration ────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("/kaggle/working/jigsaw_multicursor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_H = FRAME_W = 128       # CNN input resolution (bump to 256 on a fast GPU)
CURSOR_DIM = 5
ACT_DIM = 4

N_COLS, N_ROWS = 6, 4          # 24 pieces
PIECE_PX = 120                 # source pixels per piece (image fidelity)
CURSORS = 12                   # K cursors driven simultaneously (c < p)

N_TRAIN_ROLLOUTS = 40
N_VAL_ROLLOUTS = 6
MAX_COLLECT_STEPS = 1500
BATCH_SIZE = 64
N_EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
SNAP_POS_PX = 14.0
SNAP_ROT_DEG = 8.0
SEED = 42

print(f"Device={DEVICE}  puzzle={N_COLS}x{N_ROWS}={N_COLS*N_ROWS} pieces  K={CURSORS} cursors")

# ── Synthetic source image ───────────────────────────────────────────────────
IMG_PATH = OUT_DIR / "source.png"


def make_synthetic_image(path: Path, width: int = 1600, height: int = 1100) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]
    img = np.stack(
        [255 * xx / width, 255 * yy / height, 255 * (1.0 - (xx + yy) / (width + height))],
        axis=2,
    ).clip(0, 255).astype(np.uint8)
    for _ in range(80):
        x1, y1 = int(rng.integers(0, width - 100)), int(rng.integers(0, height - 100))
        x2 = min(x1 + int(rng.integers(60, 320)), width)
        y2 = min(y1 + int(rng.integers(60, 220)), height)
        color = rng.integers(40, 255, size=3).astype(np.float32)
        alpha = float(rng.uniform(0.2, 0.6))
        patch = img[y1:y2, x1:x2].astype(np.float32)
        img[y1:y2, x1:x2] = np.clip(patch * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


if not IMG_PATH.exists():
    make_synthetic_image(IMG_PATH)


def make_env(seed: int, cursors: int = CURSORS) -> JigsawEnvironment:
    """Fresh puzzle + scattered layout + spread-out cursor starts."""
    puzzle = generate_puzzle(str(IMG_PATH), width=N_COLS * PIECE_PX, height=N_ROWS * PIECE_PX,
                             n_cols=N_COLS, n_rows=N_ROWS, seed=seed)
    layout = shuffle_pieces(puzzle, canvas_scale=2.2, seed=seed)
    K = min(cursors, len(puzzle.pieces))
    starts = perimeter_cursor_starts(layout.canvas_width, layout.canvas_height, K)
    return JigsawEnvironment(puzzle, layout, initial_cursor_positions=starts)


# ── Model ────────────────────────────────────────────────────────────────────


class JigsawCursorNet(nn.Module):
    """One shared CNN embedding → K cursor action heads (broadcast over K)."""

    def __init__(self, cursor_dim: int = CURSOR_DIM, act_dim: int = ACT_DIM) -> None:
        super().__init__()
        # stride-2 conv stack + adaptive pool → resolution-agnostic 4x4 map
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   32,  4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.frame_proj = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
        )
        self.cursor_mlp = nn.Sequential(
            nn.Linear(cursor_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, act_dim),
        )

    def forward(self, frame: torch.Tensor, cursors: torch.Tensor) -> torch.Tensor:
        """frame (B,3,H,W), cursors (B,K,5) → (B,K,4)."""
        K = cursors.shape[1]
        frame_emb = self.frame_proj(self.cnn(frame))            # (B, 512)
        frame_exp = frame_emb.unsqueeze(1).expand(-1, K, -1)   # (B, K, 512)
        cursor_emb = self.cursor_mlp(cursors)                   # (B, K, 64)
        return self.head(torch.cat([frame_exp, cursor_emb], dim=-1))  # (B, K, 4)


# ── Dataset (frames kept uint8, normalised per-batch) ────────────────────────


class OracleDataset(Dataset):
    def __init__(self, frames: np.ndarray, cursors: np.ndarray, actions: np.ndarray) -> None:
        self.frames = torch.from_numpy(frames)                  # (T,H,W,3) uint8
        self.cursors = torch.from_numpy(cursors.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        frame = self.frames[idx].permute(2, 0, 1).float() / 255.0
        return frame, self.cursors[idx], self.actions[idx]


# ── Data collection ──────────────────────────────────────────────────────────


def _resize(frame: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).resize((FRAME_W, FRAME_H), Image.BILINEAR))


def _cursor_feat(env: JigsawEnvironment, cid: int) -> np.ndarray:
    cw, ch = env.canvas_w, env.canvas_h
    cs = env.cursors.get(cid)
    if cs is None:
        return np.zeros(CURSOR_DIM, dtype=np.float32)
    hx, hy, holding = 0.0, 0.0, 0.0
    if cs.held_piece is not None:
        c = env.piece_centroids().get(cs.held_piece)
        if c is not None:
            hx, hy, holding = c[0] / cw, c[1] / ch, 1.0
    return np.array([cs.last_x / cw, cs.last_y / ch, holding, hx, hy], dtype=np.float32)


def collect(seeds, *, cursors: int = CURSORS, max_steps: int = MAX_COLLECT_STEPS):
    """Roll out the greedy oracle and record per-step (frame, cursors, actions)."""
    frames, cur_feats, acts = [], [], []
    for seed in seeds:
        env = make_env(seed, cursors)
        env.reset()
        K = min(cursors, len(env.pieces))
        oracle = make_greedy_oracle(env, num_cursors=cursors,
                                    snap_pos_tol_px=SNAP_POS_PX, snap_rot_tol_deg=SNAP_ROT_DEG)
        for step in range(1, max_steps + 1):
            raw = env.render()
            step_actions = oracle(raw, step)
            cf = np.stack([_cursor_feat(env, c) for c in range(K)])           # (K,5)
            act = np.zeros((K, ACT_DIM), dtype=np.float32)
            act[:, 2] = -3.0
            for cid, ap in step_actions.items():
                act[cid] = [ap.x, ap.y, 3.0 if ap.grab else -3.0,
                            float(np.clip(ap.rotation_delta, -1.0, 1.0))]
            frames.append(_resize(raw))
            cur_feats.append(cf)
            acts.append(act)
            env.step(step_actions)
            if env.is_solved():
                break
        print(f"  seed {seed:3d}: {len(frames):6,d} steps total")
    return (np.stack(frames).astype(np.uint8),
            np.stack(cur_feats).astype(np.float32),
            np.stack(acts).astype(np.float32))


# ── Training ─────────────────────────────────────────────────────────────────


def _loss(pred, actions, bce):
    mse = nn.functional.mse_loss(pred[..., [0, 1, 3]], actions[..., [0, 1, 3]])
    grab = bce(pred[..., 2], (actions[..., 2] > 0).float())
    return mse + 0.5 * grab


def train(model, train_loader, val_loader, *, n_epochs=N_EPOCHS, lr=LR,
          weight_decay=WEIGHT_DECAY, pos_weight=None, device=DEVICE):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    history, best_val, best_state = [], float("inf"), None
    for epoch in range(1, n_epochs + 1):
        model.train()
        tr, n = 0.0, 0
        for frame, cur, act in train_loader:
            frame, cur, act = frame.to(device), cur.to(device), act.to(device)
            loss = _loss(model(frame, cur), act, bce)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(frame)
            n += len(frame)
        sched.step()

        model.eval()
        va, m = 0.0, 0
        with torch.no_grad():
            for frame, cur, act in val_loader:
                frame, cur, act = frame.to(device), cur.to(device), act.to(device)
                va += _loss(model(frame, cur), act, bce).item() * len(frame)
                m += len(frame)
        tr_avg, va_avg = tr / max(1, n), va / max(1, m)
        history.append({"epoch": epoch, "train": tr_avg, "val": va_avg})
        if va_avg < best_val:
            best_val = va_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == n_epochs:
            print(f"  epoch {epoch:3d}/{n_epochs}  train={tr_avg:.4f}  val={va_avg:.4f}  "
                  f"lr={sched.get_last_lr()[0]:.1e}")
    if best_state:
        model.load_state_dict(best_state)
    print(f"Best val loss: {best_val:.4f}")
    return history


# ── Inference: the model drives every cursor (no state machine) ──────────────


def make_model_policy(model, env, cursors=CURSORS, device=DEVICE):
    K = min(cursors, len(env.pieces))
    scale = 1.0 / max(1, K - 1)
    model.eval()

    def policy(obs, step):
        frame_t = torch.from_numpy(
            _resize(obs).transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(device)
        cur_t = torch.from_numpy(
            np.stack([_cursor_feat(env, c) for c in range(K)])
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(frame_t, cur_t)[0].cpu().numpy()       # (K,4)
        return {
            cid: ActionPoint(
                float(np.clip(pred[cid, 0], 0, 1)), float(np.clip(pred[cid, 1], 0, 1)),
                grab=bool(pred[cid, 2] > 0),
                rotation_delta=float(np.clip(pred[cid, 3], -1, 1)),
                render_priority=cid * scale,
            )
            for cid in range(K)
        }

    return policy


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Oracle baseline + GIF.
    print("\n=== Greedy oracle baseline ===")
    env = make_env(SEED)
    oracle = make_greedy_oracle(env, num_cursors=CURSORS,
                                snap_pos_tol_px=SNAP_POS_PX, snap_rot_tol_deg=SNAP_ROT_DEG)
    o_frames, o_res = record_rollout(env, oracle, max_steps=3000, capture_every=4,
                                     snap_to=True, snap_pos_threshold_px=SNAP_POS_PX,
                                     snap_rot_threshold_deg=SNAP_ROT_DEG)
    print(o_res.summary())
    save_gif(o_frames, OUT_DIR / "oracle_run.gif", fps=14, max_width=640)

    # 2. Collect demonstrations (separate train / val rollouts → no leakage).
    cache = OUT_DIR / "data.npz"
    if cache.exists():
        d = np.load(cache)
        trF, trC, trA = d["trF"], d["trC"], d["trA"]
        vaF, vaC, vaA = d["vaF"], d["vaC"], d["vaA"]
        print(f"\nLoaded cached data: train={len(trF):,}  val={len(vaF):,}")
    else:
        print(f"\n=== Collecting {N_TRAIN_ROLLOUTS} train + {N_VAL_ROLLOUTS} val rollouts ===")
        t0 = time.time()
        trF, trC, trA = collect(range(N_TRAIN_ROLLOUTS))
        vaF, vaC, vaA = collect(range(1000, 1000 + N_VAL_ROLLOUTS))
        np.savez_compressed(cache, trF=trF, trC=trC, trA=trA, vaF=vaF, vaC=vaC, vaA=vaA)
        print(f"Collected in {time.time()-t0:.1f}s  "
              f"(train frames {trF.nbytes/1e6:.0f} MB)")

    train_loader = DataLoader(OracleDataset(trF, trC, trA), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=DEVICE.type == "cuda")
    val_loader = DataLoader(OracleDataset(vaF, vaC, vaA), batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=DEVICE.type == "cuda")

    # 3. Train.
    model = JigsawCursorNet()
    n_params = sum(p.numel() for p in model.parameters())
    pos = (trA[..., 2] > 0).mean()
    pos_weight = torch.tensor([(1 - pos) / max(pos, 1e-3)], dtype=torch.float32)
    print(f"\n=== Training JigsawCursorNet  params={n_params:,}  grab_pos_frac={pos:.2f} ===")
    t0 = time.time()
    history = train(model, train_loader, val_loader, pos_weight=pos_weight)
    print(f"Training done in {time.time()-t0:.1f}s")
    torch.save({"state_dict": model.state_dict(),
                "config": {"frame_h": FRAME_H, "frame_w": FRAME_W, "cursors": CURSORS}},
               OUT_DIR / "jigsaw_cursor_net.pt")

    # 4. Benchmark the learned policy (it drives every cursor itself).
    print("\n=== Evaluating learned policy ===")
    env_bc = make_env(SEED)
    policy = make_model_policy(model, env_bc)
    b_frames, b_res = record_rollout(env_bc, policy, max_steps=3000, capture_every=4,
                                     snap_to=True, snap_pos_threshold_px=SNAP_POS_PX,
                                     snap_rot_threshold_deg=SNAP_ROT_DEG)
    print(b_res.summary())
    save_gif(b_frames, OUT_DIR / "bc_run.gif", fps=14, max_width=640)
    Image.fromarray(b_frames[-1]).save(OUT_DIR / "bc_final.png")

    # 5. Report.
    o, b = o_res.summary(), b_res.summary()
    print("\n┌──────────────────────────────────────────────┐")
    print(f"│  Greedy oracle   score={o['piecewise_score']:.4f}  steps={o['steps']:5d}   │")
    print(f"│  BC (PyTorch)    score={b['piecewise_score']:.4f}  steps={b['steps']:5d}   │")
    print("└──────────────────────────────────────────────┘")
    (OUT_DIR / "results.json").write_text(json.dumps(
        {"oracle": o, "bc": b, "model_params": n_params,
         "train_samples": int(len(trF)), "val_samples": int(len(vaF)),
         "epochs": N_EPOCHS, "history_tail": history[-10:]}, indent=2))
    print(f"\nOutputs in {OUT_DIR}: oracle_run.gif, bc_run.gif, jigsaw_cursor_net.pt, results.json")


if __name__ == "__main__":
    main()
