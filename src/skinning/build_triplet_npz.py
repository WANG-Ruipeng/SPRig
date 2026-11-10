#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

# --- [NEW] Canonical anchor discovery (Task 1) ---
from anchor_discovery import find_canonical_anchor, list_mesh_files, pick_neighbor_indices

# 加载你仓库的 util
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils.util import read_obj_file, read_rig_file, build_adjacency_list, compute_graph_distance

def face_areas(V, F):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

def sample_faces_by_area(F, areas, n):
    probs = areas / (areas.sum() + 1e-12)
    valid = probs > 0
    idx_valid = np.nonzero(valid)[0]
    probs = probs[valid]
    sel = np.random.choice(len(idx_valid), size=n, replace=True, p=probs).astype(np.int32)
    return idx_valid[sel]

def random_bary(n):
    u = np.random.rand(n,1); v = np.random.rand(n,1)
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]; v[mask] = 1.0 - v[mask]
    w = 1.0 - (u + v)
    return np.concatenate([u, v, w], axis=1).astype(np.float32)

def bary_points(V, F, face_ids, bary):
    f0, f1, f2 = F[face_ids,0], F[face_ids,1], F[face_ids,2]
    v0, v1, v2 = V[f0], V[f1], V[f2]
    P = (bary[:,0:1]*v0 + bary[:,1:2]*v1 + bary[:,2:3]*v2).astype(np.float32)
    fn = np.cross(v1 - v0, v2 - v0)
    N  = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    return P, N.astype(np.float32), (f0, f1, f2)

def affine_norm(V):
    bbmin, bbmax = V.min(0), V.max(0)
    c = (bbmin + bbmax) * 0.5
    s = (bbmax - bbmin).max() + 1e-8
    return c.astype(np.float32), float(s)

def apply_norm(X, c, s, factor=0.9995):
    return (X - c) / s * factor

def pack_rig(rig_path, center, scale, max_joints):
    J, bones, root_idx = read_rig_file(rig_path)
    J = (J - center) / (scale + 1e-8)
    num_j = J.shape[0]
    bone = np.zeros((num_j, 6), dtype=np.float32)
    parent = np.ones((num_j,), dtype=np.int32) * -1
    for p, c in bones:
        if parent[c] == -1: parent[c] = p
    parent[root_idx] = root_idx
    mask = parent != -1
    bone[mask,:3] = J[parent[mask]]
    bone[:,3:]    = J
    adj = build_adjacency_list(num_j, bones.astype(np.int32))
    gdist = compute_graph_distance(num_j, adj).astype(np.float32)
    pad = max(0, max_joints - num_j)
    bone_pad  = np.pad(bone,  ((0,pad),(0,0)), constant_values=0)
    gdist_pad = np.pad(gdist, ((0,pad),(0,pad)), constant_values=999)
    valid_mask = np.zeros((max_joints,), dtype=bool); valid_mask[:num_j] = True
    return bone_pad, gdist_pad, valid_mask

def pick_triplet_from_anchor(seq_dir: Path, *, criterion: str = "area"):
    """Return (A,B,C, anchor_idx, anchor_score) where A is canonical anchor."""
    # List sequence files and keep only .obj because this script uses read_obj_file
    files = [p for p in list_mesh_files(seq_dir) if p.suffix.lower() == ".obj"]
    if len(files) < 3:
        raise FileNotFoundError(f"Need >=3 .obj frames under {seq_dir}")

    anchor_idx, anchor_score, _ = find_canonical_anchor(
        seq_dir,
        criterion=criterion,
        files=files,
    )
    idxs = pick_neighbor_indices(len(files), int(anchor_idx), 3, prefer_forward=True)
    A, B, C = files[idxs[0]], files[idxs[1]], files[idxs[2]]
    return A, B, C, int(anchor_idx), float(anchor_score)

def load_split(split_file: Path):
    if not split_file.exists(): return set()
    with split_file.open("r") as f:
        return set([line.strip() for line in f if line.strip()])

def build_one(seq, dt4d_root, skel_folder, teacher_root, query_num, max_joints, *, anchor_criterion: str = "area"):
    seq_dir = dt4d_root/seq
    rig = skel_folder/f"{seq}.txt"
    # --- [Task 1] Canonical anchor discovery ---
    pa, pb, pc, anchor_idx, anchor_score = pick_triplet_from_anchor(seq_dir, criterion=anchor_criterion)

    # Teacher weights must correspond to the same canonical anchor
    tea = teacher_root/seq/f"{seq}_anchor_{anchor_idx:04d}_skin.npy"
    if not rig.exists() or not tea.exists():
        raise FileNotFoundError(f"missing rig or teacher for {seq}: {rig}, {tea}")

    # meshes
    VA, FA = read_obj_file(str(pa)); VA, FA = VA.astype(np.float32), FA.astype(np.int32)
    VB, FB = read_obj_file(str(pb)); VB, FB = VB.astype(np.float32), FB.astype(np.int32)
    VC, FC = read_obj_file(str(pc)); VC, FC = VC.astype(np.float32), FC.astype(np.int32)
    if not (FA.shape==FB.shape==FC.shape and np.array_equal(FA,FB) and np.array_equal(FA,FC)):
        raise AssertionError(f"{seq}: faces mismatch across frames")

    # sample on A
    areas = face_areas(VA, FA)
    face_ids = sample_faces_by_area(FA, areas, query_num)
    bary = random_bary(query_num)

    pA,nA,(a0,a1,a2) = bary_points(VA, FA, face_ids, bary)
    pB,nB,_ = bary_points(VB, FB, face_ids, bary)
    pC,nC,_ = bary_points(VC, FC, face_ids, bary)

    cA,sA = affine_norm(VA)
    pA = apply_norm(pA, cA, sA); pB = apply_norm(pB, cA, sA); pC = apply_norm(pC, cA, sA)

    nA = nA / (np.linalg.norm(nA, axis=1,keepdims=True)+1e-8)
    nB = nB / (np.linalg.norm(nB, axis=1,keepdims=True)+1e-8)
    nC = nC / (np.linalg.norm(nC, axis=1,keepdims=True)+1e-8)

    sample_A = np.concatenate([pA, nA], axis=1).astype(np.float32)
    sample_B = np.concatenate([pB, nB], axis=1).astype(np.float32)
    sample_C = np.concatenate([pC, nC], axis=1).astype(np.float32)

    bone_coor, graph_dist, valid_mask = pack_rig(str(rig), cA, sA, max_joints)
    Jv = int(valid_mask.sum())

    Wv = np.load(str(tea))  # (Nv, Jv)
    u,v,w = bary[:,0:1], bary[:,1:2], bary[:,2:3]
    labels_valid = (u*Wv[a0] + v*Wv[a1] + w*Wv[a2]).astype(np.float32)  # (N, Jv)

    labels = np.zeros((query_num, max_joints), dtype=np.float32)
    idx_valid = np.where(valid_mask)[0]
    labels[:, idx_valid] = labels_valid
    labels = labels / (labels.sum(axis=1, keepdims=True) + 1e-8)

    pack = dict(
        sample_A=sample_A, sample_B=sample_B, sample_C=sample_C,
        labels=labels, bone_coor=bone_coor, graph_dist=graph_dist, valid_mask=valid_mask,
        seq=np.array(seq),
        best_anchor_idx=np.int32(anchor_idx),
        best_anchor_score=np.float32(anchor_score),
        anchor_mesh_name=np.array(pa.name)
    )
    return pack

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt4d_root", type=str, default="../DeformingThings4D_objs")
    ap.add_argument("--skel_folder", type=str, default="../skeleton/DeformingThings4D_skeleton_txt")
    ap.add_argument("--teacher_root", type=str, default="./teacher_data")
    ap.add_argument("--out_root", type=str, default="./output_npz")
    ap.add_argument("--query_num", type=int, default=8192)
    ap.add_argument("--max_joints", type=int, default=70)
    ap.add_argument("--include_geometry", action="store_true", help="(不推荐大文件) 也把 vertices_*, faces 存进去，需 allow_pickle")
    ap.add_argument(
        "--anchor_criterion",
        type=str,
        default="area",
        choices=["area", "bbox_volume"],
        help="Heuristic for choosing canonical anchor per sequence.",
    )
    args = ap.parse_args()

    dt4d_root = Path(args.dt4d_root)
    skel_folder = Path(args.skel_folder)
    teacher_root = Path(args.teacher_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    split_file = skel_folder / "test_set_split.txt"
    test_set = set()
    if split_file.exists():
        with split_file.open("r") as f:
            test_set = set([x.strip() for x in f if x.strip()])

    all_seqs = [d.name for d in sorted(dt4d_root.iterdir()) if d.is_dir()]
    train_seqs = [s for s in all_seqs if s not in test_set]
    test_seqs  = [s for s in all_seqs if s in test_set]

    for subset_name, seqs in [("train", train_seqs), ("test", test_seqs)]:
        if not seqs:
            print(f"[{subset_name}] empty, skip")
            continue

        packs = []
        pbar = tqdm(seqs, desc=f"Packing {subset_name} -> big npz", unit="seq")
        for seq in pbar:
            try:
                pack = build_one(
                    seq=seq,
                    dt4d_root=dt4d_root,
                    skel_folder=skel_folder,
                    teacher_root=teacher_root,
                    query_num=args.query_num,
                    max_joints=args.max_joints,
                    anchor_criterion=args.anchor_criterion,
                )
                packs.append(pack)
            except Exception as e:
                pbar.write(f"[skip] {seq}: {e}")

        if not packs:
            print(f"[{subset_name}] nothing to save"); continue

        # 组织成大 npz：等长项 stack；可变长几何（如果需要）用 object 数组
        def stack(key, dtype_object=False):
            arrs = [p[key] for p in packs]
            if dtype_object:
                out = np.empty((len(arrs),), dtype=object)
                for i,a in enumerate(arrs): out[i] = a
                return out
            return np.stack(arrs, axis=0)

        save_dict = dict(
            sample_A = stack("sample_A"),
            sample_B = stack("sample_B"),
            sample_C = stack("sample_C"),
            labels   = stack("labels"),
            bone_coor= stack("bone_coor"),
            graph_dist=stack("graph_dist"),
            valid_mask=stack("valid_mask"),
            seq      = stack("seq")
        )

        if args.include_geometry:
            # 如果真要塞几何，这里需要再读一次 VA/VB/VC/Faces；为了省 I/O，这里不实现，
            # 建议用之前的 per-seq npz 或 HDF5 方式保存几何。
            pass

        out_path = out_root / f"{subset_name}_triplets.npz"
        np.savez_compressed(out_path, **save_dict)
        print(f"[ok] wrote {out_path}  (samples={len(packs)})")

if __name__ == "__main__":
    main()
