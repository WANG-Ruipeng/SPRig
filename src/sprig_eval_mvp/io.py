"""I/O helpers for SPRig MVP evaluation.

These helpers are intentionally self-contained so evaluation does not depend on
training-time utilities.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh


def load_mesh_vertices(path: str | Path) -> np.ndarray:
    """Load mesh vertices as (Nv,3) float32.

    Keeps original vertex ordering (process=False).
    """
    path = Path(path)
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    if not isinstance(m, trimesh.Trimesh):
        raise TypeError(f"Failed to load mesh at {path}")
    return np.asarray(m.vertices, dtype=np.float32)


def parse_rig_txt(path: str | Path) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Parse a Puppeteer-style rig text file.

    Supports both:
    - skeleton-only .txt (joints/root/hier)
    - skinning output .txt (also contains many 'skin' lines; ignored)

    Returns:
        X_rest: (K,3) float32
        parents: (K,) int64, root has parent==itself
        joint_names: list[str] length K
        root_idx: int
    """
    path = Path(path)
    joint_names: List[str] = []
    joints: List[List[float]] = []
    hier: List[Tuple[str, str]] = []  # (parent, child)
    root_name: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "joints" and len(parts) >= 5:
                joint_names.append(parts[1])
                joints.append([float(parts[2]), float(parts[3]), float(parts[4])])
            elif parts[0] == "root" and len(parts) >= 2:
                root_name = parts[1]
            elif parts[0] == "hier" and len(parts) >= 3:
                hier.append((parts[1], parts[2]))
            else:
                # ignore: skin / other metadata
                continue

    if not joints:
        raise ValueError(f"No 'joints' lines found in rig file: {path}")

    X_rest = np.asarray(joints, dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    parents = -np.ones((len(joint_names),), dtype=np.int64)
    for p_name, c_name in hier:
        if p_name in name_to_idx and c_name in name_to_idx:
            c = int(name_to_idx[c_name])
            p = int(name_to_idx[p_name])
            if parents[c] == -1:
                parents[c] = p

    if root_name is not None and root_name in name_to_idx:
        root_idx = int(name_to_idx[root_name])
    else:
        root_idx = 0

    parents[root_idx] = root_idx
    parents[parents == -1] = root_idx

    return X_rest, parents, joint_names, root_idx


def load_skinning_weights(
    npy_path: str | Path,
    *,
    fallback_joint: int = 0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Load (Nv,K) weights and sanitize them.

    - Replace NaN/Inf with 0
    - Clamp negatives to 0
    - Re-normalize rows
    - If a row sums to ~0, assign 1 to fallback_joint
    """
    W = np.load(str(npy_path))
    W = np.asarray(W, dtype=np.float32)

    if not np.isfinite(W).all():
        W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    W[W < 0.0] = 0.0
    s = W.sum(axis=1, keepdims=True)
    bad = s.squeeze(1) < eps
    if np.any(bad):
        W[bad, :] = 0.0
        fb = int(np.clip(fallback_joint, 0, W.shape[1] - 1))
        W[bad, fb] = 1.0
        s = W.sum(axis=1, keepdims=True)

    W = W / (s + eps)
    return W.astype(np.float32)


_FRAME_RE = re.compile(r"_frame_(\d+)_skin\.npy$")


def infer_anchor_from_outputs(seq_dir: Path, seq_name: str) -> Optional[int]:
    """Try to infer which anchor idx has been generated under outputs_txt/<seq>/.

    Priority:
    1) anchor_meta.json if exists
    2) parse from any '*_frame_*_skin.npy' filenames (pick the smallest idx)

    Returns:
        anchor_idx or None
    """
    meta = seq_dir / "anchor_meta.json"
    if meta.exists():
        try:
            j = json.loads(meta.read_text(encoding="utf-8"))
            if "anchor_idx" in j:
                return int(j["anchor_idx"])
        except Exception:
            pass

    cands = sorted(seq_dir.glob(f"{seq_name}_frame_*_skin.npy"))
    best = None
    for p in cands:
        m = _FRAME_RE.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        best = idx if best is None else min(best, idx)
    return best


def resolve_canonical_files(
    outputs_root: str | Path,
    seq_name: str,
    anchor_idx: int,
    *,
    strict: bool = True,
) -> Tuple[Path, Path, int]:
    """Resolve (skin_npy, rig_txt, anchor_idx_used).

    In **strict** mode (default), this enforces Task 1:
      - The outputs must exist for the computed best_anchor_idx.
      - If not, raise with a clear hint to regenerate.

    In non-strict mode, it will fall back to whatever is present.
    """
    outputs_root = Path(outputs_root)
    seq_dir = outputs_root / seq_name

    want_npy = seq_dir / f"{seq_name}_frame_{anchor_idx}_skin.npy"
    want_txt = seq_dir / f"{seq_name}_frame_{anchor_idx}_skin.txt"
    if want_npy.exists() and want_txt.exists():
        return want_npy, want_txt, int(anchor_idx)

    inferred = infer_anchor_from_outputs(seq_dir, seq_name)

    if strict:
        if inferred is None:
            raise FileNotFoundError(
                f"Expected canonical outputs for seq={seq_name} at anchor_idx={anchor_idx} but not found.\n"
                f"Missing files:\n  - {want_npy}\n  - {want_txt}\n"
                f"Hint: run skinning/bulk_generate_teachers.py after applying the patch to generate *frame_{anchor_idx}* outputs."
            )
        else:
            raise FileNotFoundError(
                f"Expected canonical outputs for seq={seq_name} at anchor_idx={anchor_idx} but found existing outputs for anchor_idx={inferred}.\n"
                f"This usually means your outputs were generated with a different anchor selection.\n"
                f"Hint: delete {seq_dir} and re-run skinning/bulk_generate_teachers.py."
            )

    # non-strict fallbacks
    if inferred is not None:
        npy2 = seq_dir / f"{seq_name}_frame_{inferred}_skin.npy"
        txt2 = seq_dir / f"{seq_name}_frame_{inferred}_skin.txt"
        if npy2.exists() and txt2.exists():
            return npy2, txt2, int(inferred)

    npys = sorted(seq_dir.glob(f"{seq_name}_*_skin.npy"))
    for npy in npys:
        txt = npy.with_suffix(".txt")
        if txt.exists():
            return npy, txt, int(anchor_idx)

    raise FileNotFoundError(f"Cannot find any skin outputs for seq={seq_name} under {seq_dir}.")
