"""DIV2K dataset → puzzle batch generation.

The DIV2K HR images live in two zips on the official mirror:
- DIV2K_train_HR.zip  (800 images, 0001..0800)
- DIV2K_valid_HR.zip  (100 images, 0801..0900)

We do not bundle the dataset; this module downloads on demand (skipped if already present)
and yields a stream of Puzzle objects so callers don't have to hold them all in memory.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from PIL import Image
from tqdm import tqdm

from .puzzle import Puzzle, generate_puzzle


DIV2K_URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


@dataclass
class DIV2KConfig:
    root: Path = Path("data/DIV2K")
    splits: tuple[str, ...] = ("train", "valid")
    download: bool = True


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dst, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _ensure_split(split: str, root: Path, download: bool) -> Path:
    folder = root / f"DIV2K_{split}_HR"
    if folder.exists() and any(folder.glob("*.png")):
        return folder
    if not download:
        raise FileNotFoundError(f"{folder} missing and download=False")
    zip_path = root / f"DIV2K_{split}_HR.zip"
    _download(DIV2K_URLS[split], zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(root)
    # The zip extracts as DIV2K_<split>_HR/*.png
    if not folder.exists():
        # Some mirrors extract differently; fall back to single-folder search.
        candidates = [p for p in root.iterdir() if p.is_dir() and split in p.name.lower()]
        if not candidates:
            raise FileNotFoundError(f"Extracted layout unrecognized at {root}")
        folder = candidates[0]
    return folder


def iter_div2k_puzzles(
    config: DIV2KConfig | None = None,
    width: int = 1200,
    height: int = 900,
    n_cols: int = 24,
    n_rows: int = 16,
    seed: int | None = 0,
    limit: int | None = None,
) -> Iterator[tuple[str, Puzzle]]:
    """Yield (image_id, Puzzle) for every image across requested DIV2K splits.

    Each puzzle uses a deterministic seed derived from `seed` and the image filename so
    runs are reproducible and parallel-friendly.
    """
    config = config or DIV2KConfig()
    config.root.mkdir(parents=True, exist_ok=True)

    count = 0
    for split in config.splits:
        folder = _ensure_split(split, config.root, config.download)
        for path in sorted(folder.glob("*.png")):
            if limit is not None and count >= limit:
                return
            image_id = path.stem
            local_seed = None if seed is None else (seed * 1_000_003 + hash(image_id)) & 0x7FFFFFFF
            with Image.open(path) as img:
                puzzle = generate_puzzle(
                    img, width=width, height=height,
                    n_cols=n_cols, n_rows=n_rows, seed=local_seed,
                )
            yield image_id, puzzle
            count += 1


def generate_div2k_puzzles_to_disk(
    out_dir: str | os.PathLike,
    config: DIV2KConfig | None = None,
    width: int = 1200,
    height: int = 900,
    n_cols: int = 24,
    n_rows: int = 16,
    seed: int | None = 0,
    limit: int | None = None,
) -> list[Path]:
    """Materialize a batch of DIV2K puzzles. Each puzzle is saved as a single .npz with
    sprites, polygons, and target metadata so downstream benchmarking can reload them
    without re-cutting."""
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for image_id, puzzle in iter_div2k_puzzles(
        config=config, width=width, height=height,
        n_cols=n_cols, n_rows=n_rows, seed=seed, limit=limit,
    ):
        sprites = np.array([p.sprite_rgba for p in puzzle.pieces], dtype=object)
        bboxes = np.array([p.bbox for p in puzzle.pieces], dtype=np.int32)
        targets = np.array([p.target_centroid for p in puzzle.pieces], dtype=np.float32)
        grid = np.array([(p.grid_row, p.grid_col) for p in puzzle.pieces], dtype=np.int32)
        out_path = out_dir / f"{image_id}.npz"
        np.savez_compressed(
            out_path,
            image=puzzle.image,
            sprites=sprites,
            bboxes=bboxes,
            targets=targets,
            grid=grid,
            width=puzzle.width, height=puzzle.height,
            n_cols=puzzle.n_cols, n_rows=puzzle.n_rows,
        )
        written.append(out_path)
    return written
