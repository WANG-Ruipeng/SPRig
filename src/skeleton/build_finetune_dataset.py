"""
Supplementary skeleton post-training dataset builder for SPRig.

This script extends the skeleton component of Seed3D/Puppeteer for
dynamic sequences. For each mesh sequence, it selects a canonical anchor
frame, predicts or reuses an anchor rig, and packages anchor-centered
keyframe point clouds into an NPZ file for fine-tuning.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh

from anchor_discovery import find_canonical_anchor, list_mesh_files, pick_neighbor_indices
from skeleton_gpt_finetune import SkeletonGPT
from utils.save_utils import (
    merge_duplicate_joints_and_fix_bones,
    pred_joints_and_bones,
    render_mesh_with_skeleton,
    save_skeleton_to_txt_joint,
)


def normalize_mesh_final(vertices, scale_factor=1.0):
    """
    Normalize vertices using the longest side of the axis-aligned bounding box.

    This follows the normalization strategy used in the official mesh-to-point
    preprocessing pipeline and provides a stable canonical scale.
    """
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)

    center = (min_bounds + max_bounds) / 2.0
    max_dimension = (max_bounds - min_bounds).max()

    if max_dimension < 1e-6:
        scale = 1.0
    else:
        scale = (2.0 * scale_factor) / max_dimension

    normalized_vertices = (vertices - center) * scale
    return normalized_vertices, center, scale


def bone_indices_to_parents(num_joints, bone_indices):
    parents = -np.ones(num_joints, dtype=np.int64)
    if not isinstance(bone_indices, (list, np.ndarray)) or len(bone_indices) == 0:
        if num_joints > 0:
            parents.fill(0)
        return parents

    for parent_idx, child_idx in bone_indices:
        if parents[child_idx] == -1:
            parents[child_idx] = parent_idx

    root_indices = np.where(parents == -1)[0]
    if len(root_indices) > 0:
        main_root_idx = root_indices[0]
        parents[root_indices] = main_root_idx
    elif num_joints > 0:
        parent_nodes = set(bone_indices[:, 0])
        child_nodes = set(bone_indices[:, 1])
        possible_roots = list(parent_nodes - child_nodes)
        if possible_roots:
            parents[possible_roots[0]] = possible_roots[0]
        else:
            parents[0] = 0
    return parents


def load_skeleton_from_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    joint_data = {}
    child_to_parent = {}
    root_name = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        keyword = parts[0]
        if keyword == 'joints':
            joint_data[parts[1]] = {'coords': np.array([float(p) for p in parts[2:]])}
        elif keyword == 'root':
            root_name = parts[1]
        elif keyword == 'hier':
            child_to_parent[parts[2]] = parts[1]

    if not joint_data or not root_name:
        return None, None

    sorted_joint_names = sorted(joint_data.keys())
    name_to_idx = {name: i for i, name in enumerate(sorted_joint_names)}
    num_joints = len(sorted_joint_names)

    joints_array = np.zeros((num_joints, 3), dtype=np.float32)
    bones_array = []

    for i, name in enumerate(sorted_joint_names):
        joints_array[i] = joint_data[name]['coords']

    for child_name, parent_name in child_to_parent.items():
        if child_name in name_to_idx and parent_name in name_to_idx:
            bones_array.append([name_to_idx[parent_name], name_to_idx[child_name]])

    return joints_array, np.array(bones_array, dtype=np.int64)


def get_args():
    parser = argparse.ArgumentParser("Fine-tuning dataset creation with optional rendering")
    parser.add_argument('--input_dir', required=True, type=str)
    parser.add_argument('--output_npz_dir', default='./DeformingThings4D_datasets', type=str)
    parser.add_argument('--output_txt_dir', default='./DeformingThings4D_skeleton_txt', type=str)
    parser.add_argument('--pretrained_weights', required=True, type=str)
    parser.add_argument('--save_render', action='store_true')
    parser.add_argument('--save_ply', action='store_true')
    parser.add_argument("--keyframe_names", nargs='+', default=['frame_0.obj', 'frame_1.obj', 'frame_2.obj'])
    parser.add_argument("--point_cloud_size", default=8192, type=int)

    parser.add_argument(
        "--anchor_criterion",
        type=str,
        default="area",
        choices=["area", "bbox_volume"],
        help="Heuristic used to choose the canonical anchor frame.",
    )
    parser.add_argument('--llm', default="facebook/opt-350m", type=str)
    parser.add_argument("--n_discrete_size", default=128, type=int)
    parser.add_argument("--n_max_bones", default=100, type=int)
    parser.add_argument(
        '--target_obj_name',
        type=str,
        default=None,
        help="If provided, only process the specified sequence.",
    )
    parser.add_argument(
        '--validation_set',
        action='store_true',
        help="If set, build the validation subset using test_set_split.txt.",
    )

    args = parser.parse_args()
    args.joint_token = True
    args.seq_shuffle = True
    args.batchsize_per_gpu = 1

    # Dataset generation is made deterministic to align with evaluation.
    args.do_sample = False
    args.num_beams = 4

    return args


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    os.makedirs(args.output_npz_dir, exist_ok=True)
    os.makedirs(args.output_txt_dir, exist_ok=True)

    if args.save_render:
        render_root_dir = os.path.join(args.output_txt_dir, 'renders')
        os.makedirs(render_root_dir, exist_ok=True)

    if args.save_ply:
        ply_root_dir = os.path.join(args.output_txt_dir, 'ply_files')
        os.makedirs(ply_root_dir, exist_ok=True)

    model = SkeletonGPT(args).to(device)
    pkg = torch.load(args.pretrained_weights, map_location="cpu")
    state_dict_to_load = pkg.get("model", pkg)
    model.load_state_dict(state_dict_to_load, strict=False)
    model.eval()

    test_split_file = os.path.join(args.input_dir, "test_set_split.txt")
    test_set_folders = set()
    if os.path.exists(test_split_file):
        with open(test_split_file, 'r') as f:
            test_set_folders = {line.strip() for line in f if line.strip()}
    elif args.validation_set:
        raise FileNotFoundError(
            f"Requested validation-set generation, but '{test_split_file}' was not found."
        )

    all_sequence_dirs = sorted(
        [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    )

    if args.validation_set:
        set_name = "validation"
        dirs_to_process = [d for d in all_sequence_dirs if d in test_set_folders]
        output_filename = "finetune_deformingthings4d_dataset_val.npz"
    else:
        set_name = "train"
        dirs_to_process = [d for d in all_sequence_dirs if d not in test_set_folders]
        output_filename = "finetune_deformingthings4d_dataset_train.npz"

    if args.target_obj_name:
        if args.target_obj_name in dirs_to_process:
            dirs_to_process = [args.target_obj_name]
        else:
            raise ValueError(
                f"Target sequence '{args.target_obj_name}' was not found in the selected '{set_name}' split."
            )

    all_processed_data = []

    for seq_name in dirs_to_process:
        try:
            seq_dir = Path(args.input_dir) / seq_name

            # Replace the hard-coded frame_0 anchor with a heuristic anchor search.
            files = list_mesh_files(seq_dir)
            best_anchor_idx, best_score, files = find_canonical_anchor(
                seq_dir,
                criterion=args.anchor_criterion,
                files=files,
            )
            anchor_mesh_path = files[best_anchor_idx]
            anchor_mesh_raw = trimesh.load(str(anchor_mesh_path), force='mesh', process=False)

            # Store a small sidecar file so that an existing rig can be reused
            # only when the selected anchor remains unchanged.
            txt_path = Path(args.output_txt_dir) / f"{seq_name}.txt"
            meta_path = Path(args.output_txt_dir) / f"{seq_name}.anchor.json"
            need_regen = True
            if txt_path.exists() and meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    need_regen = not (
                        int(meta.get("best_anchor_idx", -1)) == int(best_anchor_idx)
                        and str(meta.get("anchor_mesh_name", "")) == anchor_mesh_path.name
                        and str(meta.get("anchor_criterion", "")) == str(args.anchor_criterion)
                    )
                except Exception:
                    need_regen = True

            points_for_norm, face_indices_for_norm = anchor_mesh_raw.sample(
                args.point_cloud_size,
                return_index=True,
            )

            pc_coor_norm_for_model, master_center, master_scale = normalize_mesh_final(
                points_for_norm.copy(),
                scale_factor=0.9995,
            )

            bounds = np.array(
                [pc_coor_norm_for_model.min(axis=0), pc_coor_norm_for_model.max(axis=0)]
            )
            pc_center = (bounds[0] + bounds[1]) / 2.0
            pc_scale = (bounds[1] - bounds[0]).max()

            if txt_path.exists() and not need_regen:
                anchor_joints_denorm, anchor_bones_norm = load_skeleton_from_txt(str(txt_path))
                if anchor_joints_denorm is None:
                    continue
                temp_joints = (anchor_joints_denorm - master_center) * master_scale
                anchor_joints_norm = (temp_joints - pc_center) / pc_scale
            else:
                normals_for_model = anchor_mesh_raw.face_normals[face_indices_for_norm]

                # The generator consumes 6D point features: xyz coordinates plus normals.
                pc_for_model = np.concatenate([pc_coor_norm_for_model, normals_for_model], axis=-1)
                model_input = torch.from_numpy(pc_for_model).to(torch.float16).unsqueeze(0).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                    pred_bone_coords_norm_raw = model.generate(model_input)

                if (
                    pred_bone_coords_norm_raw is None
                    or torch.count_nonzero(~torch.isnan(pred_bone_coords_norm_raw[0, :, 0, 0])) == 0
                ):
                    continue

                joints_after_pred, bones_after_pred = pred_joints_and_bones(
                    pred_bone_coords_norm_raw[0].cpu().numpy()
                )
                joints_after_pred_B = joints_after_pred * pc_scale + pc_center
                anchor_joints_norm, anchor_bones_norm = merge_duplicate_joints_and_fix_bones(
                    joints_after_pred_B,
                    bones_after_pred,
                )
                anchor_joints_denorm = anchor_joints_norm / master_scale + master_center
                save_skeleton_to_txt_joint(anchor_joints_denorm, anchor_bones_norm, str(txt_path))

                meta_path.write_text(
                    json.dumps(
                        {
                            "best_anchor_idx": int(best_anchor_idx),
                            "best_anchor_score": float(best_score),
                            "anchor_mesh_name": anchor_mesh_path.name,
                            "anchor_criterion": str(args.anchor_criterion),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            if args.save_render:
                vertices_norm_step1 = (anchor_mesh_raw.vertices - master_center) * master_scale
                render_mesh_with_skeleton(
                    joints=anchor_joints_norm,
                    bones=anchor_bones_norm,
                    vertices=vertices_norm_step1,
                    faces=anchor_mesh_raw.faces,
                    output_dir=render_root_dir,
                    filename=seq_name,
                    prefix=f'final_puppeteer_rig_{set_name}',
                )

            anchor_parents = bone_indices_to_parents(len(anchor_joints_norm), anchor_bones_norm)

            # The anchor is always placed first, and neighboring frames provide
            # the temporal context used during fine-tuning.
            K = len(args.keyframe_names) if args.keyframe_names else 3
            keyframe_indices = pick_neighbor_indices(len(files), best_anchor_idx, K, prefer_forward=True)
            keyframe_files = [files[i] for i in keyframe_indices]

            keyframe_pointclouds = []
            for f in keyframe_files:
                mesh_raw = trimesh.load(str(f), force='mesh', process=False)
                points, face_indices = mesh_raw.sample(args.point_cloud_size, return_index=True)
                points_normalized = (points - master_center) * master_scale

                # Store 6D point clouds for every keyframe: xyz plus unit normals.
                face_normals = mesh_raw.face_normals[face_indices]
                face_normals = face_normals.astype(np.float32)
                face_normals = face_normals / (
                    np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
                )

                pc_normal = np.concatenate([points_normalized, face_normals], axis=-1).astype(np.float16)
                keyframe_pointclouds.append(pc_normal)

            sample_dict = {
                'sequence_id': seq_name,
                'keyframe_pointclouds': np.stack(keyframe_pointclouds),
                'anchor_joints': anchor_joints_norm.astype(np.float32),
                'anchor_parents': anchor_parents.astype(np.int64),
                'best_anchor_idx': int(best_anchor_idx),
                'best_anchor_score': float(best_score),
                'anchor_mesh_name': str(anchor_mesh_path.name),
                'anchor_criterion': str(args.anchor_criterion),
                'keyframe_indices': np.asarray(keyframe_indices, dtype=np.int32),
                'keyframe_names': np.asarray([p.name for p in keyframe_files], dtype=object),
            }
            all_processed_data.append(sample_dict)
        except Exception:
            continue

    if all_processed_data:
        output_npz_path = os.path.join(args.output_npz_dir, output_filename)
        np.savez_compressed(output_npz_path, sequences=np.array(all_processed_data, dtype=object))
