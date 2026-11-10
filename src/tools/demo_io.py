"""Geometry-only demo input/output checks. No model or dataset dependencies."""
from pathlib import Path
import numpy as np
import trimesh


def prepare_mesh(source, target):
    mesh = trimesh.load(source, force="mesh", process=False, skip_materials=True)
    if len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all() or mesh.extents.max() <= 0 or mesh.area <= 0:
        raise ValueError("Please provide a finite, nonempty mesh with nonzero surface area.")
    # Canonicalize OBJ syntax (triangles, positive vertex indices) for the original skin loader.
    # The returned skin array indexes THIS exported mesh, not a possibly reordered source OBJ.
    Path(target).write_text(trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_texture=False, include_color=False))
    return len(mesh.vertices), len(mesh.faces)


def validate_rig(path):
    names, joints, pairs, roots = [], [], [], []
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] == "joints":
            names.append(t[1]); joints.append([float(x) for x in t[2:5]])
        elif t[0] == "hier":
            pairs.append((t[1], t[2]))
        elif t[0] == "root":
            roots.append(t[1])
    if not 2 <= len(names) <= 70 or len(set(names)) != len(names) or len(roots) != 1 or roots[0] not in names:
        raise ValueError("Invalid skeleton: expected 2–70 distinct joints and one root.")
    if not np.isfinite(joints).all() or len(pairs) != len(names) - 1:
        raise ValueError("Skeleton must be a finite, connected tree.")
    parents, children = {}, {name: [] for name in names}
    for parent, child in pairs:
        if parent not in children or child not in children or parent == child or child in parents or child == roots[0]:
            raise ValueError("Invalid skeleton parent relation.")
        parents[child] = parent
        children[parent].append(child)
    seen, queue = set(), [roots[0]]
    while queue:
        item = queue.pop()
        if item in seen:
            raise ValueError("Skeleton contains a cycle.")
        seen.add(item); queue.extend(children[item])
    if seen != set(names):
        raise ValueError("Skeleton is disconnected; skinning was not started.")
    return len(names)


def validate_skin(path, vertex_count, joint_count):
    w = np.load(path, allow_pickle=False)
    if w.shape != (vertex_count, joint_count) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Skin array shape/values do not match the output mesh and skeleton.")
    if not np.allclose(w.sum(1), 1, atol=1e-4):
        raise ValueError("Skin weights are not normalized per vertex.")
    return list(w.shape)
