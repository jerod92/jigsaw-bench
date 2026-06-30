"""Geometric utilities for jigsaw cuts and pieces."""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Polygon


def polylines_intersect(
    a: np.ndarray,
    b: np.ndarray,
    shared_endpoints: list[np.ndarray] | None = None,
    anchor_tol_px: float = 2.0,
) -> bool:
    """True if polylines a and b cross anywhere other than near allowed shared endpoints.

    Tiny intersections within ``anchor_tol_px`` of a listed anchor are ignored — those
    are rounding artifacts from spline sampling near grid anchors and don't matter at
    raster resolution.
    """
    la = LineString(a)
    lb = LineString(b)
    inter = la.intersection(lb)
    if inter.is_empty:
        return False

    allowed = shared_endpoints or []
    pts: list[tuple[float, float]] = []
    geom_type = inter.geom_type
    if geom_type == "Point":
        pts = [(inter.x, inter.y)]
    elif geom_type == "MultiPoint":
        pts = [(p.x, p.y) for p in inter.geoms]
    elif geom_type == "GeometryCollection":
        for g in inter.geoms:
            if g.geom_type == "Point":
                pts.append((g.x, g.y))
            else:
                return True
    else:
        # LineString — real overlap
        return True

    for px, py in pts:
        if not any(np.hypot(px - ax, py - ay) < anchor_tol_px for ax, ay in allowed):
            return True
    return False


def piece_polygon(
    top: np.ndarray,
    right: np.ndarray,
    bottom: np.ndarray,
    left: np.ndarray,
) -> Polygon:
    """Build a closed piece polygon from 4 oriented edges (CCW: top L→R, right T→B, bottom R→L, left B→T)."""
    coords = []
    coords.extend(top.tolist())
    coords.extend(right.tolist()[1:])
    # bottom is given L→R, reverse for CCW traversal
    coords.extend(bottom[::-1].tolist()[1:])
    coords.extend(left[::-1].tolist()[1:-1])
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def rasterize_polygon(poly: Polygon, width: int, height: int) -> np.ndarray:
    """Boolean mask of shape (H, W) — True inside polygon. Uses PIL for speed."""
    from PIL import Image, ImageDraw

    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    if poly.geom_type == "MultiPolygon":
        polys = list(poly.geoms)
    else:
        polys = [poly]
    for p in polys:
        ext = list(p.exterior.coords)
        draw.polygon(ext, fill=1)
        for hole in p.interiors:
            draw.polygon(list(hole.coords), fill=0)
    return np.array(img, dtype=bool)


def rotate_points(pts: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate Nx2 points around a center."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return (pts - center) @ R.T + center


def aabb_overlap(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> bool:
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])
