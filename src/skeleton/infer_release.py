"""Single-mesh inference using the author's revised SPRig SkeletonGPT."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from checkpoint_io import load_strict
from skeleton_gpt_finetune import SkeletonGPT
from data_utils.save_npz import normalize_to_unit_cube
from utils.mesh_to_pc import MeshProcessor
from utils.save_utils import pred_joints_and_bones, merge_duplicate_joints_and_fix_bones, save_skeleton_to_txt_joint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--llm", required=True, help="Local, pinned OPT-350m config directory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-beams", type=int, default=4)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.n_discrete_size, a.n_max_bones, a.seq_shuffle = 128, 100, False
    a.num_beams_eval = a.num_beams
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all() or mesh.extents.max() <= 0:
        raise ValueError("Input must be a nonempty, finite mesh with nonzero extent.")
    # Same two normalizations and sampling convention as the existing demo.
    sampled = MeshProcessor.convert_meshes_to_point_clouds([mesh], 8192, apply_marching_cubes=False)[0]
    xyz, center, scale = normalize_to_unit_cube(sampled[:, :3], scale_factor=0.9995)
    pc_center = (xyz.min(0) + xyz.max(0)) / 2
    pc_scale = (xyz.max(0) - xyz.min(0)).max() + 1e-5
    normals = sampled[:, 3:]
    if not np.isfinite(xyz).all() or not (np.linalg.norm(normals, axis=-1) > .99).all():
        raise ValueError("Mesh sampling produced invalid coordinates or normals.")
    pc = torch.from_numpy(np.concatenate([xyz, normals], -1).astype(np.float32))[None].cuda()
    model = SkeletonGPT(a)
    load_strict(model, a.checkpoint)
    model.cuda().eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        prediction = model.generate(pc, deterministic=True)
    if prediction is None:
        raise RuntimeError("Skeleton generation returned no prediction.")
    edges = prediction[0].float().cpu().numpy()
    edges = edges[np.isfinite(edges).all(axis=(1, 2))]
    if not len(edges):
        raise RuntimeError("No valid skeleton edges were generated.")
    joints, bones = pred_joints_and_bones(edges)
    joints, bones = merge_duplicate_joints_and_fix_bones(joints, bones)
    if len(joints) < 2 or len(bones) == 0 or len(joints) > 70:
        raise RuntimeError(f"Generated {len(joints)} joints; the skinning release supports 2–70. No joints were silently removed.")
    joints = (joints * pc_scale + pc_center) / scale + center
    a.output.mkdir(parents=True, exist_ok=True)
    rig_path = a.output / (a.mesh.stem + ".txt")
    root = save_skeleton_to_txt_joint(joints, bones, str(rig_path))
    (a.output / "skeleton_preview.json").write_text(json.dumps({
        "joints": joints.tolist(), "bones": bones.tolist(), "root": int(root),
        "coordinate_system": "input_mesh", "num_beams": a.num_beams,
    }, indent=2))
    print(f"Skeleton saved: {rig_path}")


if __name__ == "__main__":
    main()
