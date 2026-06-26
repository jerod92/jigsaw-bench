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
    """Multi-cursor jigsaw model.

    One forward pass per step:

        frame (B, 3, H, W)  →  CNN  →  shared 512-d embedding  ─┐
                                                                  ├→  (B, K, 4) actions
        cursors (B, K, 5)   →  MLP  →  per-cursor 64-d embed  ─┘

    The frame is encoded once; that shared hidden state is broadcast to all K
    cursor heads.  nn.Linear naturally applies over the last dimension, so the
    cursor MLP and action head work identically for K=1 or K=24 without any
    reshape tricks.
    """

    def __init__(
        self,
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
            nn.AdaptiveAvgPool2d(4),
        )

        self.frame_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        # Operates on last dim → works for (B, 5) or (B, K, 5) unchanged
        self.cursor_mlp = nn.Sequential(
            nn.Linear(cursor_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # Operates on last dim → works for (B, 576) or (B, K, 576) unchanged
        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, act_dim),
        )

    def forward(self, frame: torch.Tensor, cursors: torch.Tensor) -> torch.Tensor:
        """
        frame   : (B, 3, H, W)
        cursors : (B, K, 5)
        returns : (B, K, 4)
        """
        K = cursors.shape[1]
        frame_emb = self.frame_proj(self.cnn(frame))           # (B, 512)
        frame_exp = frame_emb.unsqueeze(1).expand(-1, K, -1)  # (B, K, 512)
        cursor_emb = self.cursor_mlp(cursors)                  # (B, K, 64)
        fused = torch.cat([frame_exp, cursor_emb], dim=-1)    # (B, K, 576)
        return self.head(fused)                               # (B, K, 4)


# ── Section 5: Dataset ───────────────────────────────────────────────────────


class OracleDataset(Dataset):
    """Per-step samples: one frame + all-K cursor states + all-K actions."""

    def __init__(
        self,
        frames: np.ndarray,       # (T, H, W, 3) uint8
        cursor_feats: np.ndarray,  # (T, K, CURSOR_DIM) float32
        actions: np.ndarray,       # (T, K, ACT_DIM) float32
    ) -> None:
        frames_chw = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        self.frames = torch.from_numpy(frames_chw)
        self.cursors = torch.from_numpy(cursor_feats.astype(np.float32))
        self.actions = torch.from_numpy(actions.astype(np.float32))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        return self.frames[idx], self.cursors[idx], self.actions[idx]


# ── Section 6: Data collection ───────────────────────────────────────────────


def _resize_frame(frame: np.ndarray, h: int = FRAME_H, w: int = FRAME_W) -> np.ndarray:
    return np.array(Image.fromarray(frame).resize((w, h), Image.BILINEAR))


def collect_pytorch_data(
    puzzle_factory,
    *,
    n_rollouts: int = N_ROLLOUTS,
    num_cursors: int | None = None,
    max_steps: int = MAX_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the rule-based oracle; record one sample per step.

    Each sample contains the rendered frame plus the state and action of
    *every* cursor simultaneously — matching the model's forward signature
    (B, K, 5) → (B, K, 4).

    Returns
    -------
    frames       : (T, FRAME_H, FRAME_W, 3) uint8
    cursor_feats : (T, K, CURSOR_DIM) float32
    actions      : (T, K, ACT_DIM) float32
                     grab encoded as +3 (grab) / −3 (release) logit target
    """
    frames_list: list[np.ndarray] = []
    cursors_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []

    for rollout in range(n_rollouts):
        puzzle, layout = puzzle_factory(seed=rollout)
        env = JigsawEnvironment(puzzle, layout)
        env.reset()
        K = len(env.pieces)

        oracle, _ = _make_oracle(env, num_cursors=num_cursors)

        for step in range(1, max_steps + 1):
            raw = env.render()
            step_actions = oracle(raw, step)

            # One row per step: collect all K cursors into (K, 5) and (K, 4)
            step_cursors = np.stack(
                [build_cursor_feat(env, cid) for cid in range(K)]
            )  # (K, 5)

            step_act = np.zeros((K, ACT_DIM), dtype=np.float32)
            step_act[:, 0] = 0.5   # default: stay put, no grab
            step_act[:, 2] = -3.0
            for cid, ap in step_actions.items():
                if not ap.finished:
                    step_act[cid] = [
                        ap.x, ap.y,
                        3.0 if ap.grab else -3.0,
                        float(np.clip(ap.rotation_delta, -1.0, 1.0)),
                    ]

            frames_list.append(_resize_frame(raw))
            cursors_list.append(step_cursors)
            actions_list.append(step_act)

            env.step(step_actions)
            if env.is_solved():
                break

        print(f"  rollout {rollout + 1:3d}/{n_rollouts}  steps recorded: {len(frames_list):6,d}")

    if not frames_list:
        return (
            np.zeros((0, FRAME_H, FRAME_W, 3), dtype=np.uint8),
            np.zeros((0, N_COLS * N_ROWS, CURSOR_DIM), dtype=np.float32),
            np.zeros((0, N_COLS * N_ROWS, ACT_DIM), dtype=np.float32),
        )

    return (
        np.stack(frames_list),
        np.stack(cursors_list),
        np.stack(actions_list),
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
            pred = model(frames, cursors)              # (B, K, 4)
            loss = mse(pred[..., [0, 1, 3]], actions[..., [0, 1, 3]])
            loss = loss + 0.5 * bce(pred[..., 2], (actions[..., 2] > 0).float())
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
                pred = model(frames, cursors)              # (B, K, 4)
                loss = mse(pred[..., [0, 1, 3]], actions[..., [0, 1, 3]])
                loss = loss + 0.5 * bce(pred[..., 2], (actions[..., 2] > 0).float())
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

    One forward pass per step: frame + all-K cursor states → all-K actions.
    The state machine (piece assignment / snap detection) lives outside the
    neural net, exactly as during training.
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

        # Single forward pass: frame + all K cursor states
        small = _resize_frame(obs)
        frame_t = torch.from_numpy(
            small.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(device)                                   # (1, 3, H, W)

        cursor_np = np.stack(
            [build_cursor_feat(env, cid) for cid in range(K)]
        )                                                            # (K, 5)
        cursor_t = torch.from_numpy(cursor_np).unsqueeze(0).to(device)  # (1, K, 5)

        with torch.no_grad():
            pred = model(frame_t, cursor_t)[0].cpu().numpy()       # (K, 4)

        actions: dict[int, ActionPoint] = {}
        for cid in range(K):
            st = state[cid]
            if st["piece"] is None:
                if not queue:
                    actions[cid] = ActionPoint(0.5, 0.5, grab=False, finished=True)
                    continue
                st["piece"] = queue.pop(0)

            idx = st["piece"]
            tx, ty = targets[idx]
            pcx, pcy = centroids[idx]
            rot_err = ((rotations[idx] + 180) % 360) - 180
            if math.hypot(pcx - tx, pcy - ty) < 2.0 and abs(rot_err) < 2.0:
                actions[cid] = ActionPoint(tx / cw, ty / ch, grab=False)
                st["piece"] = None
                continue

            actions[cid] = ActionPoint(
                float(np.clip(pred[cid, 0], 0.0, 1.0)),
                float(np.clip(pred[cid, 1], 0.0, 1.0)),
                grab=bool(pred[cid, 2] > 0.0),
                rotation_delta=float(np.clip(pred[cid, 3], -1.0, 1.0)),
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
        frames_arr, cursors_arr, actions_arr = d["frames"], d["cursors"], d["actions"]
    else:
        print(f"\n=== Collecting oracle data ({N_ROLLOUTS} rollouts, K=N simultaneous) ===")
        t0 = time.time()
        frames_arr, cursors_arr, actions_arr = collect_pytorch_data(
            puzzle_factory,
            n_rollouts=N_ROLLOUTS,
            num_cursors=None,
            max_steps=MAX_STEPS,
        )
        elapsed = time.time() - t0
        frame_mb = frames_arr.nbytes / 1024 ** 2
        print(f"Collected {len(frames_arr):,} steps  frames={frame_mb:.0f} MB  "
              f"cursors={cursors_arr.shape}  in {elapsed:.1f}s")
        np.savez_compressed(data_path, frames=frames_arr, cursors=cursors_arr, actions=actions_arr)
        print(f"Saved to {data_path}")

    print(f"Steps: {len(frames_arr):,}  frames={frames_arr.shape}  "
          f"cursors={cursors_arr.shape}  actions={actions_arr.shape}")

    if len(frames_arr) == 0:
        print("No samples collected — aborting.")
        return

    # ── 3. DataLoaders ──────────────────────────────────────────────────────
    full_ds = OracleDataset(frames_arr, cursors_arr, actions_arr)
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
        "n_train_steps": int(n_train),
        "n_val_steps": int(n_val),
        "n_total_steps": int(len(frames_arr)),
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
