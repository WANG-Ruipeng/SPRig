#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finetune SkinningNet with offline A-frame teacher (triplet NPZ):
- Consistency (Symmetric KL + L1): B↔A, C↔A
- Anchor (optional): A↔A (small weight to prevent drift)
- Sharpness: entropy penalty on predictions (A/B/C)
- Bone-distance prior (optional): KL( P̃ || W· ), where P̃ ∝ exp(-β * point-to-bone distance)
- Top-K consistency masking on teacher to reduce far-joint noise
- AMP, grad accumulation, tqdm progress, prior warmup, averaged prior over A/B/C
- Best-checkpoint saving on val loss improvement + TensorBoard logging
"""

import os, sys, time, json, argparse
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.cuda.amp import GradScaler, autocast

# ============ repo imports ============
REPO_SKIN = Path(__file__).resolve().parents[0]
if str(REPO_SKIN) not in sys.path:
    sys.path.append(str(REPO_SKIN))
from skinning_models import models as skin_models  # SkinningNetStacked

# ============ Dataset ============
class TripletBigNPZ(Dataset):
    def __init__(self, npz_path, min_valid_joints=0):
        self.data = np.load(npz_path, allow_pickle=True)
        self.A = self.data["sample_A"]      # (S,N,6)
        self.B = self.data["sample_B"]
        self.C = self.data["sample_C"]
        self.labels = self.data["labels"]   # (S,N,Jmax) -> offline teacher (A)
        self.bone = self.data["bone_coor"]  # (S,Jmax,6) [parent(0:3), child(3:6)]
        self.gdist = self.data["graph_dist"]# (S,Jmax,Jmax)
        self.vmask = self.data["valid_mask"].astype(bool)  # (S,Jmax)
        self.seq = self.data["seq"].astype(str)

        if min_valid_joints > 0:
            counts = self.vmask.sum(axis=1)
            self.keep = np.where(counts >= min_valid_joints)[0]
        else:
            self.keep = np.arange(self.A.shape[0])

    def __len__(self): return len(self.keep)

    def __getitem__(self, idx):
        i = self.keep[idx]
        return dict(
            A=self.A[i], B=self.B[i], C=self.C[i],
            labels=self.labels[i],
            bone=self.bone[i], gdist=self.gdist[i], vmask=self.vmask[i],
            seq=self.seq[i]
        )

def collate(batch):
    to_t = lambda x: torch.from_numpy(np.stack(x,0)).float()
    A = to_t([b["A"] for b in batch])         # (B,N,6)
    B = to_t([b["B"] for b in batch])
    C = to_t([b["C"] for b in batch])
    labels = to_t([b["labels"] for b in batch])  # (B,N,J)
    bone = to_t([b["bone"] for b in batch])      # (B,J,6)
    gdist = to_t([b["gdist"] for b in batch])    # (B,J,J)
    vmask = torch.from_numpy(np.stack([b["vmask"] for b in batch],0))  # (B,J), bool
    seq = [b["seq"] for b in batch]
    return A,B,C,labels,bone,gdist,vmask,seq

# ============ Loss helpers ============
def mask_bnJ_from_topk(teacher, vmask, topk=None, other_scale=0.0):
    """
    teacher: (B,N,J), vmask: (B,J)
    returns m (B,N,J): valid dims mask; TopK positions = 1, other valid = other_scale; invalid = 0
    """
    B,N,J = teacher.shape
    mJ = vmask[:,None,:].expand(B,N,J)    # (B,N,J) valid joint broadcast
    if not topk or topk <= 0 or topk >= J:
        return mJ.float()

    masked = teacher.clone()
    masked[~mJ] = -1e9
    topk_idx = torch.topk(masked, k=min(topk, J), dim=-1).indices  # (B,N,K)
    m = torch.full_like(teacher, fill_value=other_scale)
    m[~mJ] = 0.0
    ar = torch.arange(B, device=teacher.device)[:,None,None]
    nr = torch.arange(N, device=teacher.device)[None,:,None]
    m[ar, nr, topk_idx] = 1.0
    return m

def renorm_over_mask(p, m, eps=1e-8):
    q = p * (m > 0).float()
    q = q / (q.sum(-1, keepdim=True) + eps)
    return q

def symm_kl(p, q, m, eps=1e-8):
    """Symmetric KL on masked dims (renormalized on mask)."""
    p = renorm_over_mask(p, m, eps)
    q = renorm_over_mask(q, m, eps)
    kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))) * m
    kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))) * m
    denom = (m.sum() / m.shape[1]) + eps  # normalize per-sample-per-point
    return (kl_pq.sum() + kl_qp.sum()) / denom

def masked_l1_bnJ(p, q, m, eps=1e-8):
    diff = (p - q).abs() * m
    denom = (m.sum() / m.shape[1]) + eps
    return diff.sum() / denom

def entropy_bnJ(p, m, eps=1e-8):
    p = renorm_over_mask(p, m, eps)
    ent = -(p * torch.log(p + eps)) * m
    denom = (m.sum() / m.shape[1]) + eps
    return ent.sum() / denom

def point_segment_distance_sq(P, A, B):
    """
    P: (B,N,3), A: (B,J,3), B: (B,J,3)
    Returns (B,N,J) squared distance from each point to segment AB.
    """
    P = P[:, :, None, :]     # (B,N,1,3)
    A = A[:, None, :, :]     # (B,1,J,3)
    B = B[:, None, :, :]     # (B,1,J,3)
    AB = B - A               # (B,1,J,3)
    AP = P - A               # (B,N,J,3)
    t = (AP * AB).sum(-1) / (AB.pow(2).sum(-1) + 1e-9)  # (B,N,J)
    t = t.clamp(0.0, 1.0)
    proj = A + t[...,None]*AB           # (B,N,J,3)
    d2 = ((P - proj).pow(2)).sum(-1)    # (B,N,J)
    return d2

def bone_prior_dist(P_xyz, bone_coor, vmask, beta=15.0):
    """
    P_xyz: (B,N,3); bone_coor: (B,J,6); vmask: (B,J)
    Return P_tilde: (B,N,J) masked+renormed prior ~ exp(-beta * dist)
    """
    parent = bone_coor[..., :3]   # (B,J,3)
    child  = bone_coor[..., 3:6]
    d2 = point_segment_distance_sq(P_xyz, parent, child)  # (B,N,J)
    d = torch.sqrt(d2 + 1e-9)
    P = torch.exp(-beta * d)
    m = vmask[:,None,:].float()
    P = P * m
    P = P / (P.sum(-1, keepdim=True) + 1e-9)
    return P

# ============ Utils ============
def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_args(args, out_dir):
    d = vars(args)
    with open(Path(out_dir) / "hparams.json", "w") as f:
        json.dump(d, f, indent=2)

# ============ Train ============
def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--train_npz", type=str, required=True)
    ap.add_argument("--test_npz",  type=str, required=True)
    ap.add_argument("--min_valid_joints", type=int, default=10)

    # model
    ap.add_argument("--pretrained_skin", type=str, default="")
    ap.add_argument("--max_joints", type=int, default=70)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--use_TAJA", action="store_true", default=True)

    # losses
    ap.add_argument("--topk", type=int, default=4, help="use top-K teacher joints for consistency; 0=off")
    ap.add_argument("--other_scale", type=float, default=0.0, help="weight for non-topK valid joints")
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_symkl", type=float, default=1.0)
    ap.add_argument("--lambda_anchor", type=float, default=0.25)
    ap.add_argument("--lambda_ent", type=float, default=0.02)
    ap.add_argument("--lambda_prior", type=float, default=0.1)
    ap.add_argument("--prior_beta", type=float, default=15.0)
    ap.add_argument("--prior_warmup_epochs", type=int, default= max(1, 2), help="linearly warm up prior to lambda_prior within this many epochs")

    # train system
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="enable torch.cuda.amp")
    ap.add_argument("--accum_steps", type=int, default=1, help="gradient accumulation steps")

    # dataloader perf
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--prefetch_factor", type=int, default=2)

    # io / logging
    ap.add_argument("--out_dir", type=str, default="./runs/skin_finetune")
    ap.add_argument("--save_ckpt", type=str, default="best.pth")
    ap.add_argument("--save_last", action="store_true", help="also save last.pth each epoch")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    # dirs
    out_dir = Path(args.out_dir)
    tb_dir = out_dir / "tb"
    ckpt_dir = out_dir / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_args(args, out_dir)

    # logging
    writer = SummaryWriter(log_dir=str(tb_dir))

    # seed / perf
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    train_ds = TripletBigNPZ(args.train_npz, min_valid_joints=args.min_valid_joints)
    test_ds  = TripletBigNPZ(args.test_npz,  min_valid_joints=args.min_valid_joints)

    coll = collate
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers>0),
        prefetch_factor=args.prefetch_factor, drop_last=True,
        collate_fn=coll
    )
    test_dl  = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers>0),
        prefetch_factor=args.prefetch_factor,
        collate_fn=coll
    )

    print(f"[data] train={len(train_ds)}  test={len(test_ds)}  "
          f"N={train_ds.A.shape[1]}  Jmax={train_ds.labels.shape[2]}")

    # model
    class MArgs: pass
    margs = MArgs()
    margs.max_joints = args.max_joints
    margs.use_TAJA = args.use_TAJA
    margs.depth = args.depth
    model = skin_models.SkinningNetStacked(margs).to(device)

    # load pretrained
    if args.pretrained_skin and os.path.exists(args.pretrained_skin):
        pkg = torch.load(args.pretrained_skin, map_location="cpu")
        model.load_state_dict(pkg["model"], strict=True)
        print(f"[model] loaded pretrained: {args.pretrained_skin}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    # training state
    global_step = 0
    best_val = float("inf")
    best_path = ckpt_dir / args.save_ckpt

    # ============ Train Loop ============
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        # prior warmup scale
        if args.lambda_prior > 0 and args.prior_warmup_epochs > 0:
            prior_scale = min(1.0, epoch / float(args.prior_warmup_epochs))
        else:
            prior_scale = 0.0 if args.lambda_prior <= 0 else 1.0
        cur_lambda_prior = args.lambda_prior * prior_scale

        # pretty header
        print(f"\nEpoch {epoch:02d}/{args.epochs} | "
              f"λ_sym={args.lambda_symkl:.2f}, λ_l1={args.lambda_l1:.2f}, "
              f"λ_anchor={args.lambda_anchor:.2f}, λ_ent={args.lambda_ent:.3f}, "
              f"λ_prior={cur_lambda_prior:.3f} (warmup to {args.lambda_prior}) | "
              f"B={args.batch_size}, AMP={'ON' if args.amp else 'OFF'}, Accum={args.accum_steps}")

        # running averages for console
        run = {"tot":0.0, "sym":0.0, "l1":0.0, "anc":0.0, "ent":0.0, "pri":0.0}
        ema = None  # for smoothing printed metrics

        optim.zero_grad(set_to_none=True)
        pbar = tqdm(enumerate(train_dl, 1), total=len(train_dl), desc=f"Train", leave=False)
        for step, (A,B,C,labels,bone,gdist,vmask,seq) in pbar:
            A,B,C = A.to(device),B.to(device),C.to(device)
            labels,bone,gdist,vmask = labels.to(device),bone.to(device),gdist.to(device),vmask.to(device)

            with autocast(enabled=args.amp):
                # forward predictions
                pred_A = model(A, bone, A, gdist, valid_mask=vmask)
                pred_B = model(B, bone, B, gdist, valid_mask=vmask)
                pred_C = model(C, bone, C, gdist, valid_mask=vmask)

                # mask for consistency (valid ∩ topK)
                m_bnJ = mask_bnJ_from_topk(labels, vmask, topk=args.topk, other_scale=args.other_scale)

                # symmetric KL + L1 (B/C ↔ A)
                loss_sym = symm_kl(labels, pred_B, m_bnJ) + symm_kl(labels, pred_C, m_bnJ)
                loss_l1  = masked_l1_bnJ(pred_B, labels, m_bnJ) + masked_l1_bnJ(pred_C, labels, m_bnJ)

                # anchor
                loss_anchor = masked_l1_bnJ(pred_A, labels, m_bnJ) if args.lambda_anchor>0 else pred_A.sum()*0

                # sharpness (entropy)
                loss_ent = entropy_bnJ(pred_A, m_bnJ) + entropy_bnJ(pred_B, m_bnJ) + entropy_bnJ(pred_C, m_bnJ)

                # bone-distance prior (three-frame averaged)
                if cur_lambda_prior > 0:
                    Ptilde_A = bone_prior_dist(A[..., :3], bone, vmask, beta=args.prior_beta)
                    Ptilde_B = bone_prior_dist(B[..., :3], bone, vmask, beta=args.prior_beta)
                    Ptilde_C = bone_prior_dist(C[..., :3], bone, vmask, beta=args.prior_beta)
                    P_tilde  = (Ptilde_A + Ptilde_B + Ptilde_C) / 3.0

                    eps=1e-8
                    p  = renorm_over_mask(P_tilde, m_bnJ)
                    qA = renorm_over_mask(pred_A,  m_bnJ)
                    qB = renorm_over_mask(pred_B,  m_bnJ)
                    qC = renorm_over_mask(pred_C,  m_bnJ)
                    denom = (m_bnJ.sum()/m_bnJ.shape[1]) + eps
                    klA = (p * (torch.log(p+eps)-torch.log(qA+eps)) * m_bnJ).sum() / denom
                    klB = (p * (torch.log(p+eps)-torch.log(qB+eps)) * m_bnJ).sum() / denom
                    klC = (p * (torch.log(p+eps)-torch.log(qC+eps)) * m_bnJ).sum() / denom
                    loss_prior = klA + klB + klC
                else:
                    loss_prior = pred_A.sum()*0

                loss = (args.lambda_symkl * loss_sym
                        + args.lambda_l1 * loss_l1
                        + args.lambda_anchor * loss_anchor
                        + args.lambda_ent * loss_ent
                        + cur_lambda_prior * loss_prior)

                # grad accumulation
                loss_acc = loss / args.accum_steps

            scaler.scale(loss_acc).backward()

            if step % args.accum_steps == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            # logging
            global_step += 1
            # exponential moving avg for console display
            comp = dict(tot=float(loss.item()), sym=float(loss_sym.item()), l1=float(loss_l1.item()),
                        anc=float(loss_anchor.item()) if args.lambda_anchor>0 else 0.0,
                        ent=float(loss_ent.item()),
                        pri=float(loss_prior.item()) if cur_lambda_prior>0 else 0.0)
            if ema is None:
                ema = comp.copy()
            else:
                for k in ema: ema[k] = 0.98*ema[k] + 0.02*comp[k]

            pbar.set_postfix({
                "tot": f"{ema['tot']:.1f}",
                "sym": f"{ema['sym']:.0f}",
                "l1":  f"{ema['l1']:.0f}",
                "anc": f"{ema['anc']:.0f}",
                "ent": f"{ema['ent']:.0f}",
                "pri": f"{ema['pri']:.0f}",
                "λp":  f"{cur_lambda_prior:.3f}",
                "B":   args.batch_size
            })

            # TensorBoard per-step (轻量一些，仅 total 和部分组件)
            if global_step % 10 == 0:
                writer.add_scalar("train/total", comp["tot"], global_step)
                writer.add_scalar("train/symmKL", comp["sym"], global_step)
                writer.add_scalar("train/L1", comp["l1"], global_step)
                writer.add_scalar("train/entropy", comp["ent"], global_step)
                if cur_lambda_prior>0:
                    writer.add_scalar("train/prior", comp["pri"], global_step)
                writer.add_scalar("train/lambda_prior_effective", cur_lambda_prior, global_step)

        # ===== Validation =====
        model.eval()
        val_l1 = 0.0
        val_sym = 0.0
        with torch.no_grad():
            pbar_v = tqdm(test_dl, total=len(test_dl), desc=f"Val  ", leave=False)
            for A,B,C,labels,bone,gdist,vmask,seq in pbar_v:
                A,B,C = A.to(device),B.to(device),C.to(device)
                labels,bone,gdist,vmask = labels.to(device),bone.to(device),gdist.to(device),vmask.to(device)

                pred_B = model(B, bone, B, gdist, valid_mask=vmask)
                pred_C = model(C, bone, C, gdist, valid_mask=vmask)
                m_bnJ  = mask_bnJ_from_topk(labels, vmask, topk=args.topk, other_scale=args.other_scale)

                val_l1 += (masked_l1_bnJ(pred_B, labels, m_bnJ).item()
                           + masked_l1_bnJ(pred_C, labels, m_bnJ).item())
                val_sym += (symm_kl(labels, pred_B, m_bnJ).item()
                           + symm_kl(labels, pred_C, m_bnJ).item())

                pbar_v.set_postfix({"val_L1(sum)": f"{val_l1:.2f}"})

        val_l1 /= max(1, len(test_dl))
        val_sym /= max(1, len(test_dl))
        epoch_time = time.time() - t0

        # epoch summary (pretty)
        header = f"Epoch {epoch:02d} done in {epoch_time:5.1f}s"
        line1  = f"  Train  | tot:{ema['tot']:10.2f}  sym:{ema['sym']:8.2f}  l1:{ema['l1']:8.2f}  anc:{ema['anc']:8.2f}  ent:{ema['ent']:8.2f}  pri:{ema['pri']:8.2f}  (λp={cur_lambda_prior:.3f})"
        line2  = f"  Valid  | L1(B,C→A):{val_l1:10.4f}  SymKL(B,C↔A):{val_sym:10.4f}"
        print(header)
        print(line1)
        print(line2)

        # TensorBoard per-epoch
        writer.add_scalar("val/L1_BC_to_A", val_l1, epoch)
        writer.add_scalar("val/SymKL_BC_to_A", val_sym, epoch)
        writer.add_scalar("meta/epoch_time_sec", epoch_time, epoch)

        # save last
        if args.save_last:
            last_path = ckpt_dir / "last.pth"
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_l1": val_l1,
                        "args": vars(args)}, last_path)

        # save best on improvement (by val L1)
        if val_l1 < best_val:
            best_val = val_l1
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_l1": val_l1,
                        "args": vars(args)}, best_path)
            print(f"[ckpt] val improved → saved best to: {best_path} (val_L1={best_val:.6f})")
            writer.add_text("ckpt/best", f"epoch={epoch}, val_L1={best_val:.6f}", epoch)

    writer.close()
    print(f"\n[done] best val_L1={best_val:.6f} | best path: {best_path}")

if __name__ == "__main__":
    main()
