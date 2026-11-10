#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ====== import model from repo ======
REPO_SKIN = Path(__file__).resolve().parents[0]
if str(REPO_SKIN) not in sys.path:
    sys.path.append(str(REPO_SKIN))
from skinning_models import models as skin_models  # SkinningNetStacked

# ====== minimal dataset (与训练一致) ======
class TripletBigNPZ(Dataset):
    def __init__(self, npz_path, min_valid_joints=0):
        self.data = np.load(npz_path, allow_pickle=True)
        self.A = self.data["sample_A"]; self.B = self.data["sample_B"]; self.C = self.data["sample_C"]
        self.labels = self.data["labels"]; self.bone = self.data["bone_coor"]
        self.gdist = self.data["graph_dist"]; self.vmask = self.data["valid_mask"].astype(bool)
        self.seq = self.data["seq"].astype(str)
        if min_valid_joints>0:
            keep = np.where(self.vmask.sum(1) >= min_valid_joints)[0]
        else:
            keep = np.arange(self.A.shape[0])
        self.keep = keep
    def __len__(self): return len(self.keep)
    def __getitem__(self, idx):
        i = self.keep[idx]
        return dict(A=self.A[i],B=self.B[i],C=self.C[i], labels=self.labels[i],
                    bone=self.bone[i], gdist=self.gdist[i], vmask=self.vmask[i], seq=self.seq[i])

def collate(batch):
    to_t = lambda xs: torch.from_numpy(np.stack(xs,0)).float()
    A=to_t([b["A"] for b in batch]); B=to_t([b["B"] for b in batch]); C=to_t([b["C"] for b in batch])
    labels=to_t([b["labels"] for b in batch]); bone=to_t([b["bone"] for b in batch])
    gdist=to_t([b["gdist"] for b in batch])
    vmask=torch.from_numpy(np.stack([b["vmask"] for b in batch],0))
    seq=[b["seq"] for b in batch]
    return A,B,C,labels,bone,gdist,vmask,seq

# ====== metrics helpers ======
def mask_topk(teacher, vmask, topk=4, other_scale=0.0):
    B,N,J = teacher.shape; mJ = vmask[:,None,:].expand(B,N,J)
    if not topk or topk>=J: return mJ.float()
    masked = teacher.clone(); masked[~mJ] = -1e9
    idx = torch.topk(masked, k=topk, dim=-1).indices
    m = torch.full_like(teacher, other_scale); m[~mJ]=0.0
    ar = torch.arange(B, device=teacher.device)[:,None,None]; nr = torch.arange(N, device=teacher.device)[None,:,None]
    m[ar,nr,idx]=1.0; return m

def renorm(p,m,eps=1e-8): q=p*(m>0); return q/(q.sum(-1,True)+eps)
def symkl(p,q,m,eps=1e-8):
    p=renorm(p,m,eps); q=renorm(q,m,eps); denom=(m.sum()/m.shape[1])+eps
    return ((p*(torch.log(p+eps)-torch.log(q+eps)))+(q*(torch.log(q+eps)-torch.log(p+eps)))*1).mul(m).sum()/denom
def l1m(p,q,m,eps=1e-8):
    denom=(m.sum()/m.shape[1])+eps; return ((p-q).abs()*m).sum()/denom
def entropy(p,m,eps=1e-8):
    p=renorm(p,m,eps); denom=(m.sum()/m.shape[1])+eps; return (-(p*torch.log(p+eps))*m).sum()/denom

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test_npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_joints", type=int, default=70)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--use_TAJA", action="store_true", default=True)
    ap.add_argument("--min_valid_joints", type=int, default=10)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--other_scale", type=float, default=0.0)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--vis_dir", type=str, default="./eval_vis")
    args=ap.parse_args()

    device="cuda" if torch.cuda.is_available() else "cpu"
    ds=TripletBigNPZ(args.test_npz, min_valid_joints=args.min_valid_joints)
    dl=DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                  pin_memory=True, collate_fn=collate)

    # model
    class M: pass
    m=M(); m.max_joints=args.max_joints; m.use_TAJA=args.use_TAJA; m.depth=args.depth
    model = skin_models.SkinningNetStacked(m).to(device).eval()
    pkg=torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(pkg["model"], strict=True)

    # run
    import matplotlib.pyplot as plt; os.makedirs(args.vis_dir, exist_ok=True)
    tot_l1=tot_sym=tot_ent=0.0; nsteps=0
    buckets = {"<20":[0,0.0], "20-40":[0,0.0], "40-70":[0,0.0]}
    with torch.no_grad():
        for A,B,C,labels,bone,gdist,vmask,seq in dl:
            A,B,C= A.to(device),B.to(device),C.to(device)
            labels,bone,gdist,vmask = labels.to(device),bone.to(device),gdist.to(device),vmask.to(device)

            pred_B = model(B,bone,B,gdist,valid_mask=vmask)
            pred_C = model(C,bone,C,gdist,valid_mask=vmask)

            m = mask_topk(labels, vmask, topk=args.topk, other_scale=args.other_scale)

            l1 = l1m(pred_B, labels, m) + l1m(pred_C, labels, m)
            sk = symkl(labels, pred_B, m) + symkl(labels, pred_C, m)
            ent = entropy(pred_B, m) + entropy(pred_C, m)

            tot_l1 += l1.item(); tot_sym += sk.item(); tot_ent += ent.item(); nsteps+=1

            # bucket by valid joints
            vcnt = vmask.sum(1).float().mean().item()
            key = "<20" if vcnt<20 else ("20-40" if vcnt<40 else "40-70")
            buckets[key][0]+=1; buckets[key][1]+=l1.item()

        print("\n== Eval on test ==")
        print(f"L1(B,C→A): {tot_l1/nsteps:.4f}  |  SymKL(B,C↔A): {tot_sym/nsteps:.4f}  |  entropy(sum): {tot_ent/nsteps:.4f}")
        for k,(c,suml1) in buckets.items():
            if c>0: print(f"  bucket {k:>5}: count={c:4d}  L1={suml1/c:.4f}")

if __name__=="__main__":
    main()
