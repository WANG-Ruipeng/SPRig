"""Canonical Anchor Discovery (Task 1).

This module provides a tiny heuristic to pick a canonical "best/most-stretched"
anchor frame from a mesh sequence.

Design goals:
- Minimal dependencies (trimesh only).
- Natural sort on filenames so frame_2 comes before frame_10.
- The returned index is the index in the natural-sorted frame list.

Outputs:
    best_anchor_idx (int): index in the sorted list
    best_score (float): surface area or bbox volume
    files (list[Path]): the sorted file list

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
import trimesh

SUPPORTED_EXTS = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}

_NUM_RE = re.compile(r"(\d+)")


def _natural_key(s: str) -> list[object]:
    """Key for natural sorting."""
    parts = _NUM_RE.split(s)
    key: list[object] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_mesh_files(mesh_dir: str | os.PathLike) -> list[Path]:
    """List mesh files under `mesh_dir` with natural-sorted filenames."""
    p = Path(mesh_dir)
    if not p.exists():
        raise FileNotFoundError(f"mesh_dir not found: {p}")
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
    files.sort(key=lambda x: _natural_key(x.name))
    if not files:
        raise FileNotFoundError(f"No mesh files found in {p} with extensions {sorted(SUPPORTED_EXTS)}")
    return files


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Failed to load mesh at {path}")
    return m


def find_canonical_anchor(
    mesh_dir: str | os.PathLike,
    *,
    criterion: Literal["area", "bbox_volume"] = "area",
    files: Sequence[Path] | None = None,
) -> Tuple[int, float, Sequence[Path]]:
    """Return (best_index, best_score, files)."""
    if files is None:
        files = list_mesh_files(mesh_dir)

    best_idx = 0
    best_score = -float("inf")

    for i, f in enumerate(files):
        m = _load_trimesh(f)
        if criterion == "area":
            score = float(m.area)
        elif criterion == "bbox_volume":
            ext = m.bounding_box.extents
            score = float(np.prod(ext))
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        if score > best_score:
            best_score = score
            best_idx = i

    return int(best_idx), float(best_score), files


def pick_neighbor_indices(
    T: int,
    anchor_idx: int,
    K: int,
    *,
    prefer_forward: bool = True,
) -> list[int]:
    """Pick K indices including anchor_idx.

    Strategy: anchor first, then fill with forward neighbors, then backward.
    Deterministic, no wrap-around.

    If T < K, will repeat deterministic fill from start.
    """

    if T <= 0:
        raise ValueError("T must be positive")
    if not (0 <= anchor_idx < T):
        raise ValueError(f"anchor_idx out of range: {anchor_idx} for T={T}")
    K = max(1, int(K))

    idxs: list[int] = [int(anchor_idx)]

    def add_if_valid(i: int):
        if 0 <= i < T and i not in idxs:
            idxs.append(i)

    if prefer_forward:
        for i in range(anchor_idx + 1, T):
            if len(idxs) >= K:
                break
            add_if_valid(i)
        for i in range(anchor_idx - 1, -1, -1):
            if len(idxs) >= K:
                break
            add_if_valid(i)
    else:
        for i in range(anchor_idx - 1, -1, -1):
            if len(idxs) >= K:
                break
            add_if_valid(i)
        for i in range(anchor_idx + 1, T):
            if len(idxs) >= K:
                break
            add_if_valid(i)

    for i in range(T):
        if len(idxs) >= K:
            break
        add_if_valid(i)

    return idxs
