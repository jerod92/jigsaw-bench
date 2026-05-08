"""Stateless CLI driver for sub-agent-as-VLM benchmarking.

By default the driver is **vision-only**: the JSON status exposes cursor position,
grab state, held piece index, solved flag, and the path to the rendered frame —
nothing the agent couldn't see by looking at the image. This is what you want for a
real VLM evaluation. Pass ``--debug`` (must come right after the subcommand) to also
dump per-piece ground truth (centroids, targets, errors) — useful for plumbing tests
or oracle agents only.

State is pickled to /tmp/jb_state.pkl; the latest frame is saved to /tmp/jb_frame.png.

Usage:
    python vlm_driver.py init [--debug]
    python vlm_driver.py move <dx> <dy> [--debug]
    python vlm_driver.py grab [--debug]
    python vlm_driver.py release [--debug]
    python vlm_driver.py rotate <degrees> [--debug]
    python vlm_driver.py status [--debug]
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from jigsaw_bench import (
    JigsawEnvironment, generate_puzzle, shuffle_pieces,
)
from jigsaw_bench.llm_interface import LLMCursorInterface


STATE = Path("/tmp/jb_state.pkl")
FRAME = Path("/tmp/jb_frame.png")
SEED = 0


def _make_image() -> Image.Image:
    W, H = 240, 180
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    quad_colors = [(255, 80, 80), (80, 200, 80), (80, 130, 255), (240, 220, 60)]
    for k, c in enumerate(quad_colors):
        qx = (k % 2) * (W // 2)
        qy = (k // 2) * (H // 2)
        d.rectangle([qx, qy, qx + W // 2, qy + H // 2], fill=c)
    d.ellipse([W // 2 - 40, H // 2 - 40, W // 2 + 40, H // 2 + 40],
              fill="white", outline="black", width=3)
    return img


def _save(state: dict, frame: np.ndarray) -> None:
    with open(STATE, "wb") as f:
        pickle.dump(state, f)
    Image.fromarray(frame).save(FRAME)


def _load() -> dict:
    with open(STATE, "rb") as f:
        return pickle.load(f)


def _status(state: dict, last_action: str, debug: bool) -> dict:
    env: JigsawEnvironment = state["env"]
    iface: LLMCursorInterface = state["iface"]
    held = env.cursors.get(iface.cursor_id)
    out = {
        "step": state["step"],
        "last_action": last_action,
        "cursor": [round(iface.cursor.x, 1), round(iface.cursor.y, 1)],
        "grabbing": iface.cursor.grabbing,
        "held_piece": held.held_piece if held else None,
        "canvas": [env.canvas_w, env.canvas_h],
        "solved": env.is_solved(pos_tol_px=12, rot_tol_deg=15),
        "frame_path": str(FRAME),
    }
    if debug:
        centroids = env.piece_centroids()
        targets = env.target_centroids()
        rotations = env.piece_rotations()
        pieces = []
        for idx in sorted(centroids):
            cx, cy = centroids[idx]
            tx, ty = targets[idx]
            rot = rotations[idx]
            rot_err = abs(((rot + 180) % 360) - 180)
            pos_err = float(np.hypot(cx - tx, cy - ty))
            pieces.append({
                "idx": idx,
                "centroid": [round(cx, 1), round(cy, 1)],
                "target": [round(tx, 1), round(ty, 1)],
                "rotation_deg": round(rot, 1),
                "pos_err_px": round(pos_err, 1),
                "rot_err_deg": round(rot_err, 1),
            })
        out["debug_pieces"] = pieces
    return out


def init(debug: bool) -> dict:
    img = _make_image()
    puzzle = generate_puzzle(img, width=240, height=180, n_cols=2, n_rows=2, seed=SEED)
    layout = shuffle_pieces(
        puzzle, canvas_scale=2.0,
        rotation_deg_choices=(0, 90, 180, 270),
        seed=SEED,
    )
    env = JigsawEnvironment(puzzle, layout)
    iface = LLMCursorInterface(
        env,
        snap_to=True,
        snap_pos_threshold_px=80,
    )
    env.reset()
    frame = iface._overlay_cursor(env.render())
    state = {"env": env, "iface": iface, "step": 0}
    _save(state, frame)
    return _status(state, "init", debug)


def step(action: str, args: list[str], debug: bool) -> dict:
    state = _load()
    iface: LLMCursorInterface = state["iface"]
    state["step"] += 1
    if action == "move":
        dx = float(args[0])
        dy = float(args[1])
        frame = iface.move(dx, dy)
        last = f"move dx={dx} dy={dy}"
    elif action == "grab":
        frame = iface.grab()
        last = "grab"
    elif action == "release":
        frame = iface.release()
        last = "release"
    elif action == "rotate":
        deg = float(args[0])
        frame = iface.rotate(deg)
        last = f"rotate {deg}"
    elif action == "status":
        env = state["env"]
        frame = iface._overlay_cursor(env.render())
        state["step"] -= 1
        last = "status"
    else:
        print(json.dumps({"error": f"unknown action {action!r}"}))
        sys.exit(2)
    _save(state, frame)
    return _status(state, last, debug)


def main() -> None:
    raw = sys.argv[1:]
    debug = "--debug" in raw
    args = [a for a in raw if a != "--debug"]
    if not args:
        print(json.dumps({"error": "missing command"}))
        sys.exit(2)
    cmd = args[0]
    if cmd == "init":
        result = init(debug)
    else:
        result = step(cmd, args[1:], debug)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
