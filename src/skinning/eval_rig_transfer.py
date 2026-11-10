#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Metric-only rig-transfer evaluation script for SPRig.

This script evaluates a fixed canonical rig and its skinning weights on a
full mesh sequence by fitting pose parameters independently for each frame
through differentiable optimization. The evaluation reports reconstruction
quality and structural stability metrics that are consistent with the
supplementary protocol.

Reported metrics:
    - LBS reconstruction RMSE (mean over frames)
    - Bone Length Variance (BLV)
    - Optional Chamfer Distance (mean over frames)

Notes:
    - Temporal jitter has been intentionally removed from this version,
      because it is not part of the paper metric suite.
    - The canonical anchor frame is selected automatically for each sequence.

Expected input layout:
    outputs_root/<sequence_name>/
        <sequence_name>_anchor_XXXX_skin.npy
        <sequence_name>_anchor_XXXX_skin.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from sprig_eval_mvp.anchor_discovery import find_canonical_anchor
from sprig_eval_mvp.pose_fitting import PoseFittingSolver, Rig
from sprig_eval_mvp.metrics import bone_length_variance, chamfer_distance, rmse as rmse_torch
from sprig_eval_mvp.io import (
    load_mesh_vertices,
    load_skinning_weights,
    parse_rig_txt,
    resolve_canonical_files,
)


def eval_one_sequence(
    *,
    seq_name: str,
    seq_dir: Path,
    outputs_root: Path,
    anchor_criterion: str,
    device: torch.device,
    num_iters: int,
    lr: float,
    optimizer: str,
    sample_verts: Optional[int],
    chamfer_samples: int,
    compute_chamfer: bool,
    warm_start: bool,
    max_frames: Optional[int],
    optimize_local_translation: bool,
) -> Dict[str, float]:
    """
    Evaluate rig transfer on a single sequence.

    The canonical anchor frame is selected first. The corresponding
    canonical skeleton and skinning weights are then loaded, and
    per-frame pose fitting is performed over the whole sequence.
    """
    anchor_list_idx, anchor_score, files = find_canonical_anchor(
        seq_dir,
        criterion=anchor_criterion,
    )

    skin_npy, rig_txt, anchor_idx_used = resolve_canonical_files(
        outputs_root,
        seq_name,
        anchor_list_idx,
    )

    anchor_mismatch = float(anchor_idx_used != anchor_list_idx)

    X_rest_np, parents_np, _, root_idx = parse_rig_txt(rig_txt)
    W_np = load_skinning_weights(skin_npy, fallback_joint=root_idx)

    V_rest_np = load_mesh_vertices(files[anchor_list_idx])

    if W_np.shape[0] != V_rest_np.shape[0]:
        raise ValueError(
            f"{seq_name}: vertex count mismatch between rest mesh "
            f"({V_rest_np.shape[0]}) and skinning weights ({W_np.shape[0]})."
        )

    if W_np.shape[1] != X_rest_np.shape[0]:
        k = min(W_np.shape[1], X_rest_np.shape[0])
        X_rest_np = X_rest_np[:k]
        parents_np = parents_np[:k]
        W_np = W_np[:, :k]

    if max_frames is not None:
        files = files[: int(max_frames)]

    X_rest = torch.from_numpy(X_rest_np).to(device=device, dtype=torch.float32)
    parents = torch.from_numpy(parents_np).to(device=device, dtype=torch.long)
    W = torch.from_numpy(W_np).to(device=device, dtype=torch.float32)
    V_rest = torch.from_numpy(V_rest_np).to(device=device, dtype=torch.float32)

    rig = Rig(X_rest=X_rest, parents=parents, W=W).to(device)
    solver = PoseFittingSolver(
        rig,
        optimize_local_translation=optimize_local_translation,
    )

    rmses: List[float] = []
    chamfers: List[float] = []
    X_worlds: List[torch.Tensor] = []

    prev_aa: Optional[torch.Tensor] = None
    prev_dt: Optional[torch.Tensor] = None

    t0 = time.time()
    for frame_idx, mesh_file in enumerate(files):
        V_t = torch.from_numpy(load_mesh_vertices(mesh_file)).to(
            device=device,
            dtype=torch.float32,
        )

        if V_t.shape[0] != V_rest.shape[0]:
            raise ValueError(
                f"{seq_name}: vertex count mismatch at frame {frame_idx} ({mesh_file.name})."
            )

        init_aa = prev_aa if warm_start else None
        init_dt = prev_dt if warm_start else None

        V_hat, X_world, _, aa, dt = solver.fit_frame(
            V_rest,
            V_t,
            sample_verts=sample_verts,
            num_iters=num_iters,
            lr=lr,
            optimizer=optimizer,
            verbose=False,
            init_aa=init_aa,
            init_dt=init_dt,
            return_params=True,
        )

        prev_aa, prev_dt = aa, dt

        rmses.append(float(rmse_torch(V_hat, V_t).item()))

        if compute_chamfer:
            cd = chamfer_distance(
                V_hat,
                V_t,
                num_samples=chamfer_samples,
            ).item()
            chamfers.append(float(cd))

        X_worlds.append(X_world.detach().cpu())

    elapsed = time.time() - t0

    X_worlds_t = torch.stack(X_worlds, dim=0).to(dtype=torch.float32)
    blv = bone_length_variance(
        X_worlds_t,
        torch.from_numpy(parents_np).long(),
    ).item()

    result: Dict[str, float] = {
        "rmse": float(np.mean(rmses)) if rmses else 0.0,
        "blv": float(blv),
        "anchor_idx": float(anchor_list_idx),
        "anchor_score": float(anchor_score),
        "anchor_mismatch": float(anchor_mismatch),
        "num_frames": float(len(files)),
        "time_sec": float(elapsed),
    }

    if compute_chamfer:
        result["chamfer"] = float(np.mean(chamfers)) if chamfers else 0.0

    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate canonical rig transfer for SPRig supplementary experiments."
    )
    ap.add_argument("--dt4d_root", type=str, required=True, help="Root directory of DeformingThings4D mesh sequences.")
    ap.add_argument("--outputs_root", type=str, required=True, help="Root directory of canonical skinning outputs.")
    ap.add_argument(
        "--anchor_criterion",
        type=str,
        default="area",
        choices=["area", "bbox_volume"],
        help="Heuristic used to select the canonical anchor frame.",
    )

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num_iters", type=int, default=60)
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Metric-only pose-fitting learning rate; this is not the Stage-2 post-training rate.",
    )
    ap.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    ap.add_argument(
        "--sample_verts",
        type=int,
        default=5000,
        help="Number of vertices subsampled during pose fitting. Set -1 to disable subsampling.",
    )
    ap.add_argument("--warm_start", action="store_true", default=True)
    ap.add_argument("--no_warm_start", action="store_true", default=False)
    ap.add_argument("--optimize_local_translation", action="store_true", default=True)
    ap.add_argument("--no_optimize_local_translation", action="store_true", default=False)

    ap.add_argument("--compute_chamfer", action="store_true", default=False)
    ap.add_argument("--chamfer_samples", type=int, default=5000)

    ap.add_argument("--max_frames", type=int, default=-1, help="Maximum number of frames per sequence. Use -1 for all frames.")
    ap.add_argument("--seq", type=str, default="", help="Evaluate a single sequence.")
    ap.add_argument("--seq_list", type=str, default="", help="Optional text file listing sequence names, one per line.")
    ap.add_argument("--out_json", type=str, required=True, help="Path to the output JSON file.")
    args = ap.parse_args()

    dt4d_root = Path(args.dt4d_root)
    outputs_root = Path(args.outputs_root)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    warm_start = bool(args.warm_start) and (not args.no_warm_start)
    optimize_local_translation = bool(args.optimize_local_translation) and (not args.no_optimize_local_translation)

    sample_verts = None if args.sample_verts < 0 else int(args.sample_verts)
    max_frames = None if args.max_frames < 0 else int(args.max_frames)

    if args.seq:
        seqs = [args.seq]
    elif args.seq_list:
        seqs = [
            s.strip()
            for s in Path(args.seq_list).read_text(encoding="utf-8").splitlines()
            if s.strip()
        ]
    else:
        dt_seqs = {d.name for d in dt4d_root.iterdir() if d.is_dir()}
        out_seqs = {d.name for d in outputs_root.iterdir() if d.is_dir()}
        seqs = sorted(list(dt_seqs & out_seqs))

    results: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}

    for seq in seqs:
        seq_dir = dt4d_root / seq
        try:
            res = eval_one_sequence(
                seq_name=seq,
                seq_dir=seq_dir,
                outputs_root=outputs_root,
                anchor_criterion=args.anchor_criterion,
                device=device,
                num_iters=int(args.num_iters),
                lr=float(args.lr),
                optimizer=args.optimizer,
                sample_verts=sample_verts,
                chamfer_samples=int(args.chamfer_samples),
                compute_chamfer=bool(args.compute_chamfer),
                warm_start=warm_start,
                max_frames=max_frames,
                optimize_local_translation=optimize_local_translation,
            )
            results[seq] = res
        except Exception as e:
            failures[seq] = repr(e)

    def mean_metric(key: str) -> float:
        vals = [results[s][key] for s in results if key in results[s]]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "num_ok": len(results),
        "num_fail": len(failures),
        "rmse_mean": mean_metric("rmse"),
        "blv_mean": mean_metric("blv"),
    }
    if args.compute_chamfer:
        summary["chamfer_mean"] = mean_metric("chamfer")

    payload = {
        "summary": summary,
        "per_sequence": results,
        "failures": failures,
        "config": {
            "dt4d_root": str(dt4d_root),
            "outputs_root": str(outputs_root),
            "anchor_criterion": args.anchor_criterion,
            "device": str(device),
            "num_iters": int(args.num_iters),
            "lr": float(args.lr),
            "optimizer": args.optimizer,
            "sample_verts": sample_verts,
            "warm_start": warm_start,
            "optimize_local_translation": optimize_local_translation,
            "max_frames": max_frames,
            "compute_chamfer": bool(args.compute_chamfer),
            "chamfer_samples": int(args.chamfer_samples),
        },
    }

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
