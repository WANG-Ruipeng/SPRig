"""End-to-end evaluation on a single mesh sequence using a fixed rig.

This script is meant as a drop-in MVP you can call from Colab while you refactor
the main repo evaluation. It assumes you already have:
- A directory of meshes (one file per frame) with consistent topology.
- A fixed rig saved as an .npz with keys: X_rest, parents, W.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh

from sprig_eval_mvp.anchor_discovery import find_canonical_anchor
from sprig_eval_mvp.pose_fitting import Rig, PoseFittingSolver
from sprig_eval_mvp.metrics import aggregate_metrics


def load_mesh_vertices(path: Path) -> np.ndarray:
    m = trimesh.load(path, force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(g for g in m.geometry.values()))
    return np.asarray(m.vertices, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_dir", type=str, required=True)
    ap.add_argument("--rig_npz", type=str, required=True)
    ap.add_argument("--criterion", type=str, default="area", choices=["area", "bbox_volume"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_iters", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    ap.add_argument("--sample_verts", type=int, default=5000)
    ap.add_argument("--chamfer_samples", type=int, default=5000)
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    # Anchor selection
    anchor_idx, score, files = find_canonical_anchor(args.mesh_dir, criterion=args.criterion)
    print(f"[anchor] idx={anchor_idx} score={score:.4f} file={files[anchor_idx].name}")

    # Load rig
    rig_data = np.load(args.rig_npz)
    X_rest = torch.from_numpy(rig_data["X_rest"]).float()
    parents = torch.from_numpy(rig_data["parents"]).long()
    W = torch.from_numpy(rig_data["W"]).float()

    # V_rest from anchor mesh
    V_rest = torch.from_numpy(load_mesh_vertices(files[anchor_idx])).float()

    # Load all target frames
    V_targets = [torch.from_numpy(load_mesh_vertices(f)).float() for f in files]
    V_targets = torch.stack(V_targets, dim=0)  # (T,Nv,3)

    rig = Rig(X_rest=X_rest, parents=parents, W=W).to(torch.device(args.device))
    V_rest = V_rest.to(args.device)
    V_targets = V_targets.to(args.device)

    solver = PoseFittingSolver(rig, optimize_local_translation=True)

    V_hats = []
    X_worlds = []
    for t in range(V_targets.shape[0]):
        V_hat, X_world, _ = solver.fit_frame(
            V_rest,
            V_targets[t],
            sample_verts=args.sample_verts,
            num_iters=args.num_iters,
            lr=args.lr,
            optimizer=args.optimizer,
            verbose=False,
        )
        V_hats.append(V_hat)
        X_worlds.append(X_world)

    V_hats = torch.stack(V_hats, dim=0)
    X_worlds = torch.stack(X_worlds, dim=0)

    m = aggregate_metrics(V_hats, V_targets, X_worlds, rig.parents, chamfer_samples=args.chamfer_samples)
    print("[metrics]", m.as_dict())

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(m.as_dict(), indent=2), encoding="utf-8")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
