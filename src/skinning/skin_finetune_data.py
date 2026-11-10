# skinning/skin_finetune_data.py
import os, glob, numpy as np, torch
import torch.utils.data as data
from utils.util import read_obj_file, read_rig_file, build_adjacency_list, compute_graph_distance, get_tpl_edges

def _face_areas(V, F):
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

def _sample_faces_by_area(F, areas, n):
    probs = areas / (areas.sum() + 1e-12)
    valid = probs > 0
    idx_valid = np.nonzero(valid)[0]
    probs = probs[valid]
    sel = np.random.choice(len(idx_valid), size=n, replace=True, p=probs).astype(np.int32)
    return idx_valid[sel]

def _rand_bary(n):
    u = np.random.rand(n, 1); v = np.random.rand(n, 1)
    mask = (u + v) > 1.0; u[mask] = 1.0 - u[mask]; v[mask] = 1.0 - v[mask]
    w = 1.0 - (u + v)
    return np.concatenate([u, v, w], axis=1).astype(np.float32)

def _bary_points(V, F, face_ids, bary, use_face_normal=True):
    f0, f1, f2 = F[face_ids,0], F[face_ids,1], F[face_ids,2]
    v0, v1, v2 = V[f0], V[f1], V[f2]
    P = (bary[:,0:1]*v0 + bary[:,1:2]*v1 + bary[:,2:3]*v2).astype(np.float32)
    if use_face_normal:
        fn = np.cross(v1 - v0, v2 - v0)
        N  = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    else:
        # 如需顶点法线，可自行加权平均实现
        fn = np.cross(v1 - v0, v2 - v0)
        N  = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-8)
    return P, N.astype(np.float32), (f0, f1, f2)

def _affine_norm(V):
    bbmin, bbmax = V.min(0), V.max(0)
    c = (bbmin + bbmax) * 0.5
    s = (bbmax - bbmin).max() + 1e-8
    return c.astype(np.float32), float(s)

def _apply_norm(X, c, s, factor=0.9995):
    return (X - c) / s * factor

class SkinFinetuneData(data.Dataset):
    """
    Triplet 数据集（A/B/C），并提供 A 帧的 Teacher 点级标签（通过顶点权重做重心插值得到）。
    需要你先离线生成 A 帧的 per-vertex Teacher：teacher_dir/<seq>_skin.npy
    """
    def __init__(self, args, root_dir, skel_dir, teacher_dir,
                 query_num=8192, use_anchor_norm=True):
        self.args = args
        self.root_dir = root_dir
        self.skel_dir = skel_dir
        self.teacher_dir = teacher_dir
        self.query_num = int(query_num)
        self.use_anchor_norm = bool(use_anchor_norm)
        self.max_joints = args.max_joints

        self.items = []  # (seq, A.obj, B.obj, C.obj, rig.txt)
        for seq in sorted(os.listdir(root_dir)):
            seq_dir = os.path.join(root_dir, seq)
            if not os.path.isdir(seq_dir): continue
            objs = sorted(glob.glob(os.path.join(seq_dir, "*.obj")))
            if len(objs) < 3: continue
            rig = os.path.join(skel_dir, f"{seq}.txt")
            tea = os.path.join(teacher_dir, f"{seq}_skin.npy")
            if not (os.path.exists(rig) and os.path.exists(tea)):
                continue
            self.items.append((seq, objs[0], objs[1], objs[2], rig, tea))
        print(f"[SkinFinetuneData] sequences: {len(self.items)}")

    def __len__(self): return len(self.items)

    @staticmethod
    def _load_mesh(p): 
        V, F = read_obj_file(p)
        return V.astype(np.float32), F.astype(np.int32)

    def _pack_rig(self, rig_path, center, scale):
        J, bones, root_idx = read_rig_file(rig_path)  # (J,3)
        J = (J - center) / (scale + 1e-8)
        num_j = J.shape[0]
        bone = np.zeros((num_j, 6), dtype=np.float32)
        parent = np.ones((num_j,), dtype=np.int32) * -1
        for p, c in bones:
            if parent[c] == -1: parent[c] = p
        parent[root_idx] = root_idx
        mask = parent != -1
        bone[mask,:3] = J[parent[mask]]
        bone[:,3:]   = J
        adj = build_adjacency_list(num_j, bones.astype(np.int32))
        gdist = compute_graph_distance(num_j, adj).astype(np.float32)
        pad = max(0, self.max_joints - num_j)
        bone_pad  = np.pad(bone, ((0,pad),(0,0)), constant_values=0)
        gdist_pad = np.pad(gdist, ((0,pad),(0,pad)), constant_values=999)
        valid_mask = np.zeros((self.max_joints,), dtype=bool); valid_mask[:num_j] = True
        return bone_pad, gdist_pad, valid_mask

    def __getitem__(self, idx):
        seq, pa, pb, pc, rig, tea = self.items[idx]
        # 读取三帧网格
        VA, FA = self._load_mesh(pa)
        VB, FB = self._load_mesh(pb)
        VC, FC = self._load_mesh(pc)
        # 拓扑一致性
        if not (FA.shape == FB.shape == FC.shape and np.array_equal(FA, FB) and np.array_equal(FA, FC)):
            raise AssertionError(f"[{seq}] A/B/C faces 不一致，无法做 (face,bary) 对应")

        # A 帧面面积采样 + 重心
        areas = _face_areas(VA, FA)
        face_ids = _sample_faces_by_area(FA, areas, self.query_num)      # (N,)
        bary     = _rand_bary(self.query_num)                             # (N,3)
        pA, nA, (a0,a1,a2) = _bary_points(VA, FA, face_ids, bary)
        pB, nB, _           = _bary_points(VB, FB, face_ids, bary)
        pC, nC, _           = _bary_points(VC, FC, face_ids, bary)

        # 统一归一化（以 A 帧为基准）
        cA, sA = _affine_norm(VA)
        pA = _apply_norm(pA, cA, sA); pB = _apply_norm(pB, cA, sA); pC = _apply_norm(pC, cA, sA)
        VA_ = _apply_norm(VA, cA, sA); VB_ = _apply_norm(VB, cA, sA); VC_ = _apply_norm(VC, cA, sA)

        # 组装三帧的 (N,6)
        qa = np.concatenate([pA, nA / (np.linalg.norm(nA,axis=1,keepdims=True)+1e-8)], axis=1).astype(np.float32)
        qb = np.concatenate([pB, nB / (np.linalg.norm(nB,axis=1,keepdims=True)+1e-8)], axis=1).astype(np.float32)
        qc = np.concatenate([pC, nC / (np.linalg.norm(nC,axis=1,keepdims=True)+1e-8)], axis=1).astype(np.float32)

        # 骨架打包（同一 rig，用 A 的坐标系）
        bone_pad, gdist_pad, valid_mask = self._pack_rig(rig, cA, sA)

        # Edges（可用于平滑正则/可视化）
        EA = get_tpl_edges(VA_, FA).astype(np.int64)
        EB = get_tpl_edges(VB_, FB).astype(np.int64)
        EC = get_tpl_edges(VC_, FC).astype(np.int64)

        # ---------- Teacher: 顶点权重 -> 采样点重心插值 ----------
        Wv = np.load(tea)  # 形状 (Nv, N_valid_joints)
        # 注意：Wv 对应的是 valid joints 的子集列。需要扩到 max_joints。
        num_j_valid = int(valid_mask.sum())
        assert Wv.shape[1] == num_j_valid, f"{seq}: Teacher 列数与 valid joints 不符"
        # 取该面三个顶点的权重，并按 bary 插值
        Wa = Wv[a0]; Wb = Wv[a1]; Wc = Wv[a2]                      # (N, Jv)
        u, v, w = bary[:,0:1], bary[:,1:2], bary[:,2:3]
        Wp_valid = (u*Wa + v*Wb + w*Wc).astype(np.float32)          # (N, Jv)
        # 扩到 max_joints 并归一化
        Wp = np.zeros((self.query_num, self.max_joints), dtype=np.float32)
        idx_valid = np.nonzero(valid_mask)[0]
        Wp[:, idx_valid] = Wp_valid
        Wp = Wp / (Wp.sum(axis=1, keepdims=True) + 1e-8)

        # ---------- 打包返回 ----------
        A_tuple = (torch.from_numpy(qa), torch.from_numpy(qa),
                   torch.from_numpy(bone_pad), torch.from_numpy(valid_mask),
                   torch.from_numpy(gdist_pad),
                   torch.from_numpy(VA_).float(), f"{seq}_A", torch.from_numpy(EA))
        B_tuple = (torch.from_numpy(qb), torch.from_numpy(qb),
                   torch.from_numpy(bone_pad), torch.from_numpy(valid_mask),
                   torch.from_numpy(gdist_pad),
                   torch.from_numpy(VB_).float(), f"{seq}_B", torch.from_numpy(EB))
        C_tuple = (torch.from_numpy(qc), torch.from_numpy(qc),
                   torch.from_numpy(bone_pad), torch.from_numpy(valid_mask),
                   torch.from_numpy(gdist_pad),
                   torch.from_numpy(VC_).float(), f"{seq}_C", torch.from_numpy(EC))

        labels = torch.from_numpy(Wp)  # (N, max_joints), A 帧 teacher 点级标签
        meta = {"seq": seq, "face_ids": torch.from_numpy(face_ids), "bary": torch.from_numpy(bary)}

        # 训练时你会用：W_A_pred、W_B_pred、W_C_pred 与 labels 计算损失（见下一节）
        return A_tuple, B_tuple, C_tuple, labels, meta
