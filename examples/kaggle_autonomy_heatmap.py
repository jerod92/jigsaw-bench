#!/usr/bin/env python3
"""Kaggle GPU experiment: does data scale crack the approach-assignment wall?

This is the decisive test from examples/MULTICURSOR_BC_FINDINGS.md. A fully
autonomous multi-cursor policy (no oracle, no controller at inference) drives
every cursor through approach -> grab -> carry -> derotate -> release in one
forward pass per step. Each sub-skill was solved on CPU *except* approach target
localization ("which scattered piece is mine?"), which is multimodal: the
factorized x/y head invents ghost modes, and the joint 2D heatmap is the
ghost-free fix but was data-starved on a CPU budget.

Here we give the joint 2D heatmap what it needs: thousands of training puzzles,
128px frames, and a real held-out validation set. If the autonomous solve rate
climbs, data was the lever; if approach still fails, the bottleneck is
assignment ambiguity (-> RL with piecewise_score, or slot-attention).

Requirements
------------
- Kaggle accelerator: GPU (T4/P100).
- Internet: ON (for git clone).

Run::

    !python /kaggle/working/jigsaw-bench/examples/kaggle_autonomy_heatmap.py

Knobs are CONFIG constants below. Defaults target a 2x2 puzzle (cleanest test of
approach assignment); bump N_COLS/N_ROWS once approach works.
"""
from __future__ import annotations

# ── bootstrap: clone + install ───────────────────────────────────────────────
import subprocess
import sys
from pathlib import Path

REPO = Path("/kaggle/working/jigsaw-bench")
if not REPO.exists():
    print("Cloning jigsaw-bench …")
    subprocess.run(["git", "clone", "https://github.com/jerod92/jigsaw-bench.git", str(REPO)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(REPO), "-q"], check=True)
sys.path.insert(0, str(REPO))

# ── imports ──────────────────────────────────────────────────────────────────
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from jigsaw_bench import (
    ActionPoint, JigsawEnvironment, benchmark_model, generate_puzzle,
    make_greedy_oracle, perimeter_cursor_starts, record_rollout, save_gif, shuffle_pieces,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("/kaggle/working/autonomy"); OUT.mkdir(parents=True, exist_ok=True)

N_COLS, N_ROWS = 2, 2          # puzzle grid (start at 2x2 for the cleanest test)
PIECE_PX = 130                 # source pixels per piece
FRAME = 128                    # CNN input resolution
NBINS = 64                     # joint move heatmap is NBINS x NBINS
CURSORS = N_COLS * N_ROWS      # one cursor per piece

N_TRAIN_SEEDS = 3000           # <-- the lever: far more puzzles than the CPU runs
N_VAL_SEEDS = 200              # real held-out val set for honest early stopping
COLLECT_SNAP = 4.0             # oracle places to ~4px so demos cover full convergence
EVAL_SNAP = 12.0               # snap-to assist at eval (snaps to EXACT pose)
BATCH = 256
EPOCHS = 60
LR = 1e-3
SEED = 0

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"device={DEVICE}  puzzle={N_COLS}x{N_ROWS}={CURSORS} cursors  frame={FRAME}  bins={NBINS}")

# ── source image (quadrant-tinted gradient: pieces are visually identifiable) ──
IMG = OUT / "source.png"
if not IMG.exists():
    w = h = 600
    yy, xx = np.mgrid[0:h, 0:w]
    im = np.stack([255*xx/w, 255*yy/h, 255*(1-(xx+yy)/(w+h))], 2).clip(0, 255).astype(np.uint8)
    for qy, qx, col in [(0,0,(220,40,40)),(0,1,(40,220,40)),(1,0,(40,40,220)),(1,1,(220,220,40))]:
        ys, xs = slice(qy*h//2,(qy+1)*h//2), slice(qx*w//2,(qx+1)*w//2)
        im[ys, xs] = (im[ys, xs]*0.3 + np.array(col)*0.7).astype(np.uint8)
    Image.fromarray(im).save(IMG)


def make_env(seed):
    p = generate_puzzle(str(IMG), width=N_COLS*PIECE_PX, height=N_ROWS*PIECE_PX,
                        n_cols=N_COLS, n_rows=N_ROWS, seed=seed)
    lay = shuffle_pieces(p, canvas_scale=2.2, seed=seed)
    starts = perimeter_cursor_starts(lay.canvas_width, lay.canvas_height, CURSORS)
    return JigsawEnvironment(p, lay, initial_cursor_positions=starts)


def frame_in(raw, env):
    """Render with each cursor drawn as a white ring so 'am I on a piece' is a
    local figure-ground cue (this is what made the grab head learnable)."""
    img = Image.fromarray(raw).convert("RGB")
    d = ImageDraw.Draw(img)
    r = max(12, int(0.022 * max(env.canvas_w, env.canvas_h)))
    for cid in range(CURSORS):
        cs = env.cursors.get(cid)
        if cs is None:
            continue
        x, y = cs.last_x, cs.last_y
        d.ellipse([x-r, y-r, x+r, y+r], outline=(255, 255, 255), width=max(3, r // 4))
        d.ellipse([x-3, y-3, x+3, y+3], fill=(255, 255, 255))
    return np.asarray(img.resize((FRAME, FRAME), Image.BILINEAR))


def cfeat(env, cid):
    cw, ch = env.canvas_w, env.canvas_h
    cs = env.cursors.get(cid)
    if cs is None:
        return np.zeros(5, np.float32)
    hx, hy, hold = 0.0, 0.0, 0.0
    if cs.held_piece is not None:
        c = env.piece_centroids().get(cs.held_piece)
        if c:
            hx, hy, hold = c[0]/cw, c[1]/ch, 1.0
    return np.array([cs.last_x/cw, cs.last_y/ch, hold, hx, hy], np.float32)


def _rot_err(deg):
    return ((deg + 180.0) % 360.0) - 180.0


# ── model: shared CNN feature map, sampled per cursor; joint 2D heatmap move ──
class MultiCursorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),    # 128->64
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),   # 64->32
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True), # 32->16
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),# 16x16, keep spatial
        )
        self.global_proj = nn.Sequential(nn.Linear(128, 192), nn.ReLU(True))
        self.state_mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(True))
        trunk_in = 128 + 192 + 64
        self.trunk = nn.Sequential(nn.Linear(trunk_in, 256), nn.ReLU(True), nn.Dropout(0.3),
                                   nn.Linear(256, 192), nn.ReLU(True), nn.Dropout(0.3))
        self.heat_head = nn.Linear(192, NBINS * NBINS)   # joint 2D — no ghost cross terms
        self.res_head = nn.Linear(192, 2)                # within-cell residual
        self.grab_head = nn.Linear(192, 1)
        self.rot_head = nn.Linear(192, 1)
        self.home_head = nn.Linear(192, 2)               # auxiliary

    def forward(self, frame, cur):
        B, K, _ = cur.shape
        feat = self.cnn(frame)                                   # (B,C,h,w)
        g = self.global_proj(feat.mean((2, 3)))                  # (B,192)
        grid = (cur[..., :2] * 2 - 1).unsqueeze(2)               # (B,K,1,2)
        local = F.grid_sample(feat, grid, align_corners=True).squeeze(-1).permute(0, 2, 1)  # (B,K,C)
        s = self.state_mlp(cur)
        h = self.trunk(torch.cat([local, g.unsqueeze(1).expand(-1, K, -1), s], -1))
        return (self.heat_head(h), torch.tanh(self.res_head(h)) * 0.5,
                self.grab_head(h).squeeze(-1), self.rot_head(h).squeeze(-1), self.home_head(h))


# ── data collection: ABSOLUTE destination labels (self-correcting policy) ─────
def collect(seeds):
    Fr, Cu, Mv, Gr, Ro, Ho, Ms = [], [], [], [], [], [], []
    for s in seeds:
        try:
            env = make_env(s)
        except RuntimeError:
            continue
        env.reset()
        orc = make_greedy_oracle(env, num_cursors=CURSORS, snap_pos_tol_px=COLLECT_SNAP, snap_rot_tol_deg=2.0)
        tgt = env.target_centroids(); cw, ch = env.canvas_w, env.canvas_h
        corner = (cw*0.97, ch*0.97); last_held = {}
        for step in range(1, 260):
            raw = env.render(); acts = orc(raw, step); rots = env.piece_rotations()
            cf = np.stack([cfeat(env, c) for c in range(CURSORS)])
            mv = np.zeros((CURSORS, 2), np.float32); gr = np.zeros(CURSORS, np.float32)
            ro = np.zeros(CURSORS, np.float32); ho = np.zeros((CURSORS, 2), np.float32)
            am = np.zeros(CURSORS, np.float32); hm = np.zeros(CURSORS, np.float32)
            for c in range(CURSORS):
                ap = acts.get(c)
                if ap is None or ap.finished:
                    continue
                am[c] = 1.0; gr[c] = 1.0 if ap.grab else 0.0
                cs = env.cursors.get(c)
                if cs is not None and cs.held_piece is not None:           # CARRY: dest = home+offset
                    p = cs.held_piece; tx, ty = tgt[p]
                    ox, oy = cs.grab_offset_local if cs.grab_offset_local else (0.0, 0.0)
                    mv[c] = [(tx+ox)/cw, (ty+oy)/ch]; ro[c] = float(np.clip(-_rot_err(rots[p])/180.0, -1, 1))
                    hm[c] = 1.0; ho[c] = [tx/cw, ty/ch]
                else:
                    piece = orc.cursor[c].get("piece")
                    if piece is not None:                                   # APPROACH: dest = grab point
                        ix, iy = orc._interior_canvas(piece); mv[c] = [ix/cw, iy/ch]
                    else:                                                   # IDLE: corner
                        mv[c] = [corner[0]/cw, corner[1]/ch]
            Fr.append(frame_in(raw, env)); Cu.append(cf); Mv.append(mv)
            Gr.append(gr); Ro.append(ro); Ho.append(ho); Ms.append(np.stack([am, hm], 0))
            env.step(acts)
            now = {cid: c.held_piece for cid, c in env.cursors.items()}
            for cid, prev in list(last_held.items()):
                if prev is not None and prev != now.get(cid):
                    env.snap_piece(prev, COLLECT_SNAP, 2.0)
            last_held = now
            if env.is_solved():
                break
    return [np.stack(x) for x in (Fr, Cu, Mv, Gr, Ro, Ho, Ms)]


class DS(Dataset):
    def __init__(self, *a):
        self.d = [torch.from_numpy(x) for x in a]

    def __len__(self):
        return len(self.d[0])

    def __getitem__(self, i):
        Fr, Cu, Mv, Gr, Ro, Ho, Ms = (x[i] for x in self.d)
        return (Fr.permute(2, 0, 1).float()/255., Cu.float(), Mv.float(),
                Gr.float(), Ro.float(), Ho.float(), Ms.float())


def policy(model, env):
    model.eval()

    def pol(obs, step):
        ft = torch.from_numpy(frame_in(obs, env).transpose(2, 0, 1).astype(np.float32)/255.).unsqueeze(0).to(DEVICE)
        ct = torch.from_numpy(np.stack([cfeat(env, c) for c in range(CURSORS)])).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            heat, res, gr, ro, _ = model(ft, ct)
        flat = heat[0].argmax(-1).cpu().numpy()
        xb, yb = flat % NBINS, flat // NBINS
        dx, dy = res[0, :, 0].cpu().numpy(), res[0, :, 1].cpu().numpy()
        gr, ro = gr[0].cpu().numpy(), ro[0].cpu().numpy()
        return {c: ActionPoint(float(np.clip((xb[c]+0.5+dx[c])/NBINS, 0, 1)),
                               float(np.clip((yb[c]+0.5+dy[c])/NBINS, 0, 1)),
                               grab=bool(gr[c] > 0), rotation_delta=float(np.clip(ro[c], -1, 1)),
                               render_priority=c/max(1, CURSORS-1)) for c in range(CURSORS)}
    return pol


def main():
    cache = OUT / "data.npz"
    if cache.exists():
        d = np.load(cache)
        tr = [d[f"tr{i}"] for i in range(7)]; va = [d[f"va{i}"] for i in range(7)]
        print(f"loaded cache: train={len(tr[0])} val={len(va[0])}")
    else:
        t0 = time.time()
        print(f"collecting {N_TRAIN_SEEDS} train + {N_VAL_SEEDS} val puzzles …")
        tr = collect(range(N_TRAIN_SEEDS))
        va = collect(range(100000, 100000 + N_VAL_SEEDS))
        np.savez_compressed(cache, **{f"tr{i}": tr[i] for i in range(7)},
                            **{f"va{i}": va[i] for i in range(7)})
        print(f"collected train={len(tr[0])} val={len(va[0])} in {time.time()-t0:.0f}s")

    tl = DataLoader(DS(*tr), batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=DEVICE.type == 'cuda')
    vl = DataLoader(DS(*va), batch_size=BATCH, num_workers=2, pin_memory=DEVICE.type == 'cuda')
    model = MultiCursorNet().to(DEVICE)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    pos = float(tr[3][tr[6][:, 0] > 0].mean())
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1-pos)/max(pos, 1e-3)]).to(DEVICE), reduction='none')

    def losses(batch):
        fr, cu, mv, gr, ro, ho, ms = (x.to(DEVICE) for x in batch)
        B, K = ms[:, 0].shape
        am, hm = ms[:, 0], ms[:, 1]
        heat, res, pgr, pro, pho = model(fr, cu)
        xb = (mv[..., 0].clamp(0, 0.9999)*NBINS).long(); yb = (mv[..., 1].clamp(0, 0.9999)*NBINS).long()
        cls = (yb*NBINS + xb).reshape(B*K)
        lheat = (F.cross_entropy(heat.reshape(B*K, NBINS*NBINS), cls, reduction='none')*am.reshape(B*K)).sum()/am.sum().clamp(min=1)
        rtgt = torch.stack([mv[..., 0]*NBINS - xb - 0.5, mv[..., 1]*NBINS - yb - 0.5], -1)
        lres = (((res-rtgt)**2).mean(-1)*am).sum()/am.sum().clamp(min=1)
        lgr = (bce(pgr, gr)*am).sum()/am.sum().clamp(min=1)
        lro = (((pro-ro)**2)*hm).sum()/hm.sum().clamp(min=1)
        lho = ((((pho-ho)**2).mean(-1))*hm).sum()/hm.sum().clamp(min=1)
        return 0.3*lheat + lres + 1.5*lgr + 0.5*lro + 0.3*lho, lheat.item(), lgr.item()

    best, best_sd = 1e9, None
    for ep in range(1, EPOCHS+1):
        model.train()
        for b in tl:
            loss, _, _ = losses(b)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); v = vh = vg = 0.0; n = 0
        with torch.no_grad():
            for b in vl:
                lo, lh, lg = losses(b); bs = len(b[0]); v += lo.item()*bs; vh += lh*bs; vg += lg*bs; n += bs
        v, vh, vg = v/n, vh/n, vg/n
        if v < best:
            best = v; best_sd = {k: val.cpu().clone() for k, val in model.state_dict().items()}
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d}/{EPOCHS}  val={v:.4f}  heatCE={vh:.3f}  grabBCE={vg:.3f}")
    model.load_state_dict(best_sd)
    torch.save(model.state_dict(), OUT / "autonomy.pt")
    print(f"best val={best:.4f}")

    # ── decisive metric: autonomous solve rate on held-out puzzles ──
    print("\n=== Autonomous evaluation (model drives everything) ===")
    solved, scores = 0, []
    for i, seed in enumerate(range(200000, 200020)):
        try:
            env = make_env(seed)
        except RuntimeError:
            continue
        res = benchmark_model(env, policy(model, env), max_steps=200, snap_to=True,
                              snap_pos_threshold_px=EVAL_SNAP, snap_rot_threshold_deg=EVAL_SNAP*0.6)
        solved += int(res.solved); scores.append(res.piecewise_score)
        if i < 4:
            frames, _ = record_rollout(env, policy(model, env), max_steps=200, capture_every=3,
                                       snap_to=True, snap_pos_threshold_px=EVAL_SNAP,
                                       snap_rot_threshold_deg=EVAL_SNAP*0.6)
            save_gif(frames, OUT / f"autonomy_{seed}.gif", fps=14, max_width=480)
    print(f"\nAUTONOMOUS SOLVED {solved}/{len(scores)}  mean_score={np.mean(scores):.3f}")
    (OUT / "results.json").write_text(json.dumps(
        {"solved": solved, "n": len(scores), "mean_score": float(np.mean(scores)),
         "best_val": best, "train_samples": int(len(tr[0])), "config": {
             "grid": [N_COLS, N_ROWS], "frame": FRAME, "nbins": NBINS,
             "train_seeds": N_TRAIN_SEEDS, "epochs": EPOCHS}}, indent=2))
    print(f"outputs in {OUT}")


if __name__ == "__main__":
    main()
