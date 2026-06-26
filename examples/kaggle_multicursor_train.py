#!/usr/bin/env python3
"""Kaggle training script: multi-cursor behavioral cloning for jigsaw-bench.

Clones the repo, collects oracle demonstrations, trains a PyTorch CNN model,
and evaluates it against the rule-based oracle baseline.

Requirements
------------
- Kaggle accelerator: GPU T4 or P100
- Internet access: ON (Notebook Settings → Internet → On)

Run from a Kaggle notebook cell::

    !python /kaggle/working/jigsaw-bench/examples/kaggle_multicursor_train.py

Or paste sections as notebook cells (section markers make clean split points).

Architecture: JigsawCursorNet
------------------------------
Frame encoder : (B, 3, 64, 64)  →  4-layer stride-2 CNN  →  Linear(4096, 512)
Cursor MLP    : (B, 5)           →  Linear(5, 64) → ReLU → Linear(64, 64)
Fusion head   : (B, 576)         →  Linear(576, 256) → ReLU → Linear(256, 128)
                                     → ReLU → Linear(128, 4)

Output (BC_ACT_DIM = 4)
  [0]  action_x        normalised target x ∈ [0, 1]
  [1]  action_y        normalised target y ∈ [0, 1]
  [2]  grab_logit      sigmoid > 0.5 → grab=True
  [3]  rotation_delta  ∈ [−1, 1] → (−180°, +180°) per step

Piece coordinates are intentionally absent from the model's inputs; the CNN
must infer piece locations from the rendered frame, exactly as a human would.
"""
from __future__ import annotations

# ── Section 0: Setup ─────────────────────────────────────────────────────────
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

# Make the examples directory importable for _make_oracle / build_cursor_feat
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "examples"))

# ── Section 1: Imports ───────────────────────────────────────────────────────
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from jigsaw_bench import (
    ActionPoint,
    JigsawEnvironment,
    generate_puzzle,
    record_rollout,
    save_gif,
    shuffle_pieces,
)
from geo_bc_oracle import _make_oracle, build_cursor_feat, make_rule_oracle

# ── Section 2: Configuration ─────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("/kaggle/working/jigsaw_multicursor")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_H: int = 128   # CNN input height  (bump to 256 on GPU for better quality)
FRAME_W: int = 128   # CNN input width
CURSOR_DIM: int = 5  # GEO_CURSOR_DIM
ACT_DIM: int = 4     # (action_x, action_y, grab_logit, rotation_delta)

PUZZLE_W: int = 900
PUZZLE_H: int = 600
N_COLS: int = 6
N_ROWS: int = 4      # 24 pieces total

N_ROLLOUTS: int = 40
MAX_STEPS: int = 1200
BATCH_SIZE: int = 128
N_EPOCHS: int = 150
LR: float = 3e-4
WEIGHT_DECAY: float = 1e-4
VAL_FRACTION: float = 0.15
SEED: int = 42

print(f"Device : {DEVICE}")
print(f"Puzzle : {N_COLS}×{N_ROWS} = {N_COLS * N_ROWS} pieces")
print(f"Output : {OUT_DIR}")

# ── Section 3: Synthetic source image ────────────────────────────────────────
IMG_PATH = OUT_DIR / "puzzle_source.png"


def make_synthetic_image(path: Path, width: int = 1200, height: int = 800) -> None:
    """Create a colourful procedural image so no real photo is needed."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]
    r = (255 * xx / width).astype(np.float32)
    g = (255 * yy / height).astype(np.float32)
    b = 255 * (1.0 - (xx + yy) / (width + height)).astype(np.float32)
    img = np.stack([r, g, b], axis=2).clip(0, 255).astype(np.uint8)

    for _ in range(50):
        x1 = int(rng.integers(0, width - 80))
        y1 = int(rng.integers(0, height - 80))
        x2 = min(x1 + int(rng.integers(80, 350)), width)
        y2 = min(y1 + int(rng.integers(80, 220)), height)
        color = rng.integers(40, 255, size=3).astype(np.float32)
        alpha = float(rng.uniform(0.25, 0.65))
        patch = img[y1:y2, x1:x2].astype(np.float32)
        img[y1:y2, x1:x2] = np.clip(patch * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)

    Image.fromarray(img).save(path)
    print(f"Synthetic image: {path}  ({width}×{height} px)")


if not IMG_PATH.exists():
    make_synthetic_image(IMG_PATH)

# ── Section 4: Model architecture ────────────────────────────────────────────


class JigsawCursorNet(nn.Module):
    """Multi-cursor jigsaw model: CNN frame encoder + cursor MLP + fusion head.

    Inputs
    ------
    frame  : (B, 3, FRAME_H, FRAME_W) float32 normalised to [0, 1]
    cursor : (B, CURSOR_DIM) float32
               [x/W, y/H, is_holding, held_cx/W, held_cy/H]

    Output
    ------
    actions : (B, ACT_DIM) float32
                [action_x, action_y, grab_logit, rotation_delta]
    """

    def __init__(
        self,
        frame_h: int = FRAME_H,
        frame_w: int = FRAME_W,
        cursor_dim: int = CURSOR_DIM,
        act_dim: int = ACT_DIM,
    ) -> None:
        super().__init__()

        # 5-layer stride-2 CNN + adaptive pool → resolution-agnostic 4×4 output
        # 128×128 → 64 → 32 → 16 → 8 → 4 → AdaptivePool → 4×4  (same for 256×256)
        self.cnn = nn.Sequential(
            nn.Conv2d(3,   32,  kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True),
            nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # always 4×4 regardless of input resolution
        )
        cnn_flat = 256 * 4 * 4  # = 4096, resolution-agnostic

        self.frame_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_flat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.cursor_mlp = nn.Sequential(
            nn.Linear(cursor_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, act_dim),
        )

    def forward(self, frame: torch.Tensor, cursor: torch.Tensor) -> torch.Tensor:
        frame_feat = self.frame_proj(self.cnn(frame))          # (B, 512)
        cursor_feat = self.cursor_mlp(cursor)                  # (B, 64)
        return self.head(torch.cat([frame_feat, cursor_feat], dim=1))  # (B, 4)


# ── Section 5: Dataset ───────────────────────────────────────────────────────


class OracleDataset(Dataset):
    """(frame, cursor_feat, action) triples collected from rule-based oracle.

    Frames are stored deduplicated (one per step); each sample holds an index
    into the frame array rather than its own copy.  With K=24 cursors this
    saves ~24× memory vs naïvely repeating the frame per cursor.
    """

    def __init__(
        self,
        frames: np.ndarray,        # (F, H, W, 3) uint8 — unique frames
        frame_indices: np.ndarray,  # (N,) int32      — maps sample → frame row
        cursor_feats: np.ndarray,   # (N, CURSOR_DIM) float32
        actions: np.ndarray,        # (N, ACT_DIM) float32
    ) -> None:
        # HWC uint8 → CHW float32 normalised to [0, 1]
        frames_chw = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        self.frames = torch.from_numpy(frames_chw)
        self.frame_indices = torch.from_numpy(frame_indices.astype(np.int64))
        self.cursors = torch.from_numpy(cursor_feats.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, idx: int):
        return self.frames[self.frame_indices[idx]], self.cursors[idx], self.actions[idx]


# ── Section 6: Data collection ───────────────────────────────────────────────


def _resize_frame(frame: np.ndarray, h: int = FRAME_H, w: int = FRAME_W) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize((w, h), Image.BILINEAR))


def collect_pytorch_data(
    puzzle_factory,
    *,
    n_rollouts: int = N_ROLLOUTS,
    num_cursors: int | None = None,
    max_steps: int = MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the rule-based oracle and record (frame, cursor_feat, action) triples.

    Frames are stored once per step and shared across all K cursors active at
    that step.  With K=24 this cuts frame memory ~24× compared to duplicating.

    Returns
    -------
    frames        : (F, FRAME_H, FRAME_W, 3) uint8  — unique frames (one/step)
    frame_indices : (N,) int32                       — maps sample i → frames row
    cursor_feats  : (N, CURSOR_DIM) float32
    actions       : (N, ACT_DIM) float32
                      grab column encoded as +3 (grab) / −3 (release) logit
    """
    unique_frames: list[np.ndarray] = []  # one per step across all rollouts
    frame_indices: list[int] = []
    cursor_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []

    for rollout in range(n_rollouts):
        puzzle, layout = puzzle_factory(seed=rollout)
        env = JigsawEnvironment(puzzle, layout)
        env.reset()

        oracle, _ = _make_oracle(env, num_cursors=num_cursors)

        for step in range(1, max_steps + 1):
            raw = env.render()
            small = _resize_frame(raw)  # (FRAME_H, FRAME_W, 3) uint8

            actions = oracle(raw, step)

            # Store this frame once; all cursors in this step share the index
            frame_idx = len(unique_frames)
            any_active = any(not ap.finished for ap in actions.values())
            if any_active:
                unique_frames.append(small)

            for cid, ap in actions.items():
                if ap.finished:
                    continue
                c_feat = build_cursor_feat(env, cid)
                act = np.array(
                    [ap.x, ap.y, 3.0 if ap.grab else -3.0,
                     float(np.clip(ap.rotation_delta, -1.0, 1.0))],
                    dtype=np.float32,
                )
                frame_indices.append(frame_idx)
                cursor_list.append(c_feat)
                action_list.append(act)

            env.step(actions)
            if env.is_solved():
                break

        n_samples = len(frame_indices)
        n_frames = len(unique_frames)
        print(f"  rollout {rollout + 1:3d}/{n_rollouts}  samples: {n_samples:7,d}  unique frames: {n_frames:5,d}")

    if not unique_frames:
        return (
            np.zeros((0, FRAME_H, FRAME_W, 3), dtype=np.uint8),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, CURSOR_DIM), dtype=np.float32),
            np.zeros((0, ACT_DIM), dtype=np.float32),
        )

    return (
        np.stack(unique_frames),
        np.array(frame_indices, dtype=np.int32),
        np.stack(cursor_list),
        np.stack(action_list),
    )


# ── Section 7: Training loop ─────────────────────────────────────────────────


def train_model(
    model: JigsawCursorNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    device: torch.device = DEVICE,
) -> list[dict]:
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    history: list[dict] = []
    best_val = float("inf")
    best_state: dict | None = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        tr_loss, n_tr = 0.0, 0
        for frames, cursors, actions in train_loader:
            frames, cursors, actions = frames.to(device), cursors.to(device), actions.to(device)
            pred = model(frames, cursors)
            # Position + rotation: MSE on cols [0, 1, 3]
            loss = mse(pred[:, [0, 1, 3]], actions[:, [0, 1, 3]])
            # Grab: BCE on col [2]
            loss = loss + 0.5 * bce(pred[:, 2], (actions[:, 2] > 0).float())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(frames)
            n_tr += len(frames)
        scheduler.step()

        model.eval()
        va_loss, n_va = 0.0, 0
        with torch.no_grad():
            for frames, cursors, actions in val_loader:
                frames, cursors, actions = frames.to(device), cursors.to(device), actions.to(device)
                pred = model(frames, cursors)
                loss = mse(pred[:, [0, 1, 3]], actions[:, [0, 1, 3]])
                loss = loss + 0.5 * bce(pred[:, 2], (actions[:, 2] > 0).float())
                va_loss += loss.item() * len(frames)
                n_va += len(frames)

        tr_avg = tr_loss / max(1, n_tr)
        va_avg = va_loss / max(1, n_va)
        history.append({"epoch": epoch, "train": tr_avg, "val": va_avg})

        if va_avg < best_val:
            best_val = va_avg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 15 == 0 or epoch == n_epochs:
            print(f"  epoch {epoch:4d}/{n_epochs}  train={tr_avg:.4f}  val={va_avg:.4f}"
                  f"  lr={scheduler.get_last_lr()[0]:.1e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val loss: {best_val:.4f}")
    return history


# ── Section 8: Inference wrapper ─────────────────────────────────────────────


def make_torch_oracle(
    model: JigsawCursorNet,
    env: JigsawEnvironment,
    num_cursors: int | None = None,
    device: torch.device = DEVICE,
):
    """Wrap a trained JigsawCursorNet in the piece-assignment state machine.

    Maintains the same FIFO queue / phase logic as the rule-based oracle so
    the neural net only needs to predict raw actions, not manage assignments.
    """
    N = len(env.pieces)
    K = N if (num_cursors is None or num_cursors <= 0) else min(num_cursors, N)

    targets = env.target_centroids()
    queue: list[int] = list(targets.keys())
    state: dict[int, dict] = {cid: {"piece": None} for cid in range(K)}
    render_scale = 1.0 / max(1, K - 1) if K > 1 else 1.0

    model.eval()

    def oracle(obs: np.ndarray, step: int) -> dict[int, ActionPoint]:
        cw, ch = env.canvas_w, env.canvas_h
        centroids = env.piece_centroids()
        rotations = env.piece_rotations()

        # Prepare frame tensor once — shared across all cursors this step
        small = _resize_frame(obs)
        frame_t = torch.from_numpy(
            small.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(device)  # (1, 3, H, W)

        actions: dict[int, ActionPoint] = {}

        for cid, st in state.items():
            if st["piece"] is None:
                if not queue:
                    actions[cid] = ActionPoint(0.5, 0.5, grab=False, finished=True)
                    continue
                st["piece"] = queue.pop(0)

            idx = st["piece"]
            c_feat = build_cursor_feat(env, cid)
            cursor_t = torch.from_numpy(c_feat).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(frame_t, cursor_t)[0].cpu().numpy()  # (4,)

            ax = float(np.clip(pred[0], 0.0, 1.0))
            ay = float(np.clip(pred[1], 0.0, 1.0))
            grab = bool(pred[2] > 0.0)
            rot_delta = float(np.clip(pred[3], -1.0, 1.0))

            # Retire piece once within snap distance of its target
            tx, ty = targets[idx]
            pcx, pcy = centroids[idx]
            rot_err = ((rotations[idx] + 180) % 360) - 180
            if math.hypot(pcx - tx, pcy - ty) < 2.0 and abs(rot_err) < 2.0:
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                continue

            actions[cid] = ActionPoint(
                ax, ay, grab=grab,
                rotation_delta=rot_delta,
                render_priority=cid * render_scale,
            )

        return actions

    return oracle


# ── Section 9: Main ──────────────────────────────────────────────────────────


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    def puzzle_factory(seed: int = 0):
        puzzle = generate_puzzle(
            str(IMG_PATH), width=PUZZLE_W, height=PUZZLE_H,
            n_cols=N_COLS, n_rows=N_ROWS, seed=seed,
        )
        layout = shuffle_pieces(puzzle, canvas_scale=2.5, seed=seed)
        return puzzle, layout

    # ── 1. Rule-based oracle baseline ──────────────────────────────────────
    print("\n=== Rule-based oracle (baseline) ===")
    puzzle, layout = puzzle_factory(seed=SEED)
    env_rb = JigsawEnvironment(puzzle, layout)
    rb_model = make_rule_oracle(env_rb, num_cursors=None)  # K = N = 24 cursors
    rb_frames, rb_result = record_rollout(
        env_rb, rb_model, max_steps=2000, capture_every=8,
        snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
    )
    print(rb_result.summary())
    Image.fromarray(rb_frames[-1]).save(OUT_DIR / "oracle_final.png")
    save_gif(rb_frames, OUT_DIR / "oracle_run.gif", fps=12, max_width=640)

    # ── 2. Collect training data ────────────────────────────────────────────
    data_path = OUT_DIR / "oracle_data.npz"
    if data_path.exists():
        print(f"\n=== Loading cached data from {data_path} ===")
        d = np.load(data_path)
        frames_arr = d["frames"]
        frame_idx_arr = d["frame_indices"]
        cursors_arr, actions_arr = d["cursors"], d["actions"]
    else:
        print(f"\n=== Collecting oracle data ({N_ROLLOUTS} rollouts, K=N simultaneous) ===")
        t0 = time.time()
        frames_arr, frame_idx_arr, cursors_arr, actions_arr = collect_pytorch_data(
            puzzle_factory,
            n_rollouts=N_ROLLOUTS,
            num_cursors=None,
            max_steps=MAX_STEPS,
        )
        elapsed = time.time() - t0
        frame_mb = frames_arr.nbytes / 1024 ** 2
        print(f"Collected {len(frame_idx_arr):,} samples ({len(frames_arr):,} unique frames, "
              f"{frame_mb:.0f} MB) in {elapsed:.1f}s")
        np.savez_compressed(
            data_path,
            frames=frames_arr, frame_indices=frame_idx_arr,
            cursors=cursors_arr, actions=actions_arr,
        )
        print(f"Saved to {data_path}")

    print(f"Unique frames : {frames_arr.shape}  ({frames_arr.nbytes // 1024**2} MB)")
    print(f"Samples       : {len(frame_idx_arr):,}  cursors={cursors_arr.shape}  actions={actions_arr.shape}")

    if len(frame_idx_arr) == 0:
        print("No samples collected — aborting.")
        return

    # ── 3. DataLoaders ──────────────────────────────────────────────────────
    full_ds = OracleDataset(frames_arr, frame_idx_arr, cursors_arr, actions_arr)
    n_val = max(1, int(len(full_ds) * VAL_FRACTION))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=DEVICE.type == "cuda",
    )
    print(f"\nTrain: {n_train:,}  Val: {n_val:,}  Batches/epoch: {len(train_loader)}")

    # ── 4. Train ────────────────────────────────────────────────────────────
    model = JigsawCursorNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training JigsawCursorNet  params={n_params:,}  device={DEVICE} ===")

    t0 = time.time()
    history = train_model(model, train_loader, val_loader)
    print(f"Training done in {time.time() - t0:.1f}s")

    weights_path = OUT_DIR / "jigsaw_cursor_net.pt"
    torch.save({"state_dict": model.state_dict(), "config": {
        "frame_h": FRAME_H, "frame_w": FRAME_W,
        "cursor_dim": CURSOR_DIM, "act_dim": ACT_DIM,
    }}, weights_path)
    print(f"Weights saved: {weights_path}")

    # ── 5. Evaluate trained model ───────────────────────────────────────────
    print("\n=== Evaluating trained BC model ===")
    puzzle, layout = puzzle_factory(seed=SEED)
    env_bc = JigsawEnvironment(puzzle, layout)
    torch_oracle = make_torch_oracle(model, env_bc, num_cursors=None)
    bc_frames, bc_result = record_rollout(
        env_bc, torch_oracle, max_steps=2000, capture_every=8,
        snap_to=True, snap_pos_threshold_px=18, snap_rot_threshold_deg=10,
    )
    print(bc_result.summary())
    Image.fromarray(bc_frames[-1]).save(OUT_DIR / "bc_final.png")
    save_gif(bc_frames, OUT_DIR / "bc_run.gif", fps=12, max_width=640)

    # ── 6. Results ──────────────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────┐")
    print("│                    Results Summary                  │")
    print("├─────────────────────────────────────────────────────┤")
    rb = rb_result.summary()
    bc = bc_result.summary()
    print(f"│  Rule-based oracle   score={rb['piecewise_score']:.4f}  steps={rb['steps']:5d}   │")
    print(f"│  BC (PyTorch CNN)    score={bc['piecewise_score']:.4f}  steps={bc['steps']:5d}   │")
    print("└─────────────────────────────────────────────────────┘")

    results = {
        "rule_based": rb,
        "bc_cnn": bc,
        "n_train_samples": int(n_train),
        "n_val_samples": int(n_val),
        "n_unique_frames": int(len(frames_arr)),
        "n_epochs": N_EPOCHS,
        "model_params": n_params,
        "device": str(DEVICE),
        "training_history": history[-10:],  # last 10 epochs
    }
    results_path = OUT_DIR / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nAll outputs saved to: {OUT_DIR}")
    print(f"  oracle_data.npz      — raw training data")
    print(f"  jigsaw_cursor_net.pt — trained model weights")
    print(f"  oracle_run.gif       — rule-based oracle full rollout")
    print(f"  bc_run.gif           — BC model full rollout")
    print(f"  oracle_final.png     — rule-based oracle final frame")
    print(f"  bc_final.png         — BC model final frame")
    print(f"  results.json         — benchmark comparison")


if __name__ == "__main__":
    main()
