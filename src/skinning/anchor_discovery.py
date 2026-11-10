"""
Canonical anchor discovery utility for SPRig.

This module provides a lightweight heuristic to select a canonical anchor
frame from a mesh sequence. The anchor is chosen as the frame that best
matches a simple geometric criterion, such as maximum surface area or
maximum axis-aligned bounding-box volume.
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
    """
    Natural-sort key so that numerical suffixes are ordered numerically.

    For example, frame_2 is sorted before frame_10.
    """
    parts = _NUM_RE.split(s)
    key: list[object] = []
    for p in parts:
        if p.isdigit():
            key.append(int(p))
        else:
            key.append(p.lower())
    return key


def list_mesh_files(mesh_dir: str | os.PathLike) -> list[Path]:
    """
    List mesh files in a directory using natural filename ordering.
    """
    p = Path(mesh_dir)
    if not p.exists():
        raise FileNotFoundError(f"Mesh directory not found: {p}")

    files = [
        f
        for f in p.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]
    files.sort(key=lambda x: _natural_key(x.name))

    if not files:
        raise FileNotFoundError(
            f"No supported mesh files were found in {p}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTS)}"
        )
    return files


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    """
    Load a mesh file as a trimesh.Trimesh object.
    """
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Failed to load mesh from {path}")
    return mesh


def find_canonical_anchor(
    mesh_dir: str | os.PathLike,
    *,
    criterion: Literal["area", "bbox_volume"] = "area",
    files: Sequence[Path] | None = None,
) -> Tuple[int, float, Sequence[Path]]:
    """
    Select the canonical anchor frame from a mesh sequence.

    Args:
        mesh_dir: Directory containing a mesh-per-frame sequence.
        criterion: Geometric criterion used for anchor selection.
            - "area": maximum surface area
            - "bbox_volume": maximum axis-aligned bounding-box volume
        files: Optional pre-listed mesh files.

    Returns:
        best_idx: Index of the selected anchor frame.
        best_score: Heuristic score of the selected frame.
        files: Ordered mesh-file sequence used for evaluation.
    """
    if files is None:
        files = list_mesh_files(mesh_dir)

    best_idx = 0
    best_score = -float("inf")

    for i, f in enumerate(files):
        mesh = _load_trimesh(f)

        if criterion == "area":
            score = float(mesh.area)
        elif criterion == "bbox_volume":
            extents = mesh.bounding_box.extents
            score = float(np.prod(extents))
        else:
            raise ValueError(f"Unknown anchor criterion: {criterion}")

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score, files


def pick_neighbor_indices(
    T: int,
    anchor_idx: int,
    K: int,
    *,
    prefer_forward: bool = True,
) -> list[int]:
    """
    Select K frame indices centered on a canonical anchor.

    The returned list always starts with the anchor frame. Remaining indices
    are filled deterministically by scanning forward first and then backward,
    or vice versa if `prefer_forward=False`. No wrap-around is used.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (0 <= anchor_idx < T):
        raise ValueError(f"anchor_idx={anchor_idx} is out of range for T={T}.")

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
