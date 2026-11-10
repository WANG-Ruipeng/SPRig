"""Differentiable pose fitting (IK-style) solver for SPRig evaluation.

MVP goals:
- Accept a *fixed* canonical rig (joint rest positions + parents + skinning weights).
- Optimize per-frame joint SE(3) parameters to reconstruct a target mesh via LBS.
- Return reconstructed vertices and fitted joint world positions for downstream metrics.

Notes
-----
1) This is an *evaluation-time* solver, not part of the backbone. It is safe to
   evolve independently from the training code.
2) We optimize **local rotations + local translations** along the kinematic tree.
   - If local translations are enabled, bone lengths are allowed to vary, making
     BLV meaningful.
3) Supports vertex subsampling during optimization (IMPORTANT for speed).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import axis_angle_to_matrix, make_transform, invert_transform


def _topo_order(parents: torch.Tensor) -> torch.Tensor:
    """Return a topological order for a rooted tree given parent indices."""
    K = parents.numel()
    children = [[] for _ in range(K)]
    roots = []
    for j in range(K):
        p = int(parents[j])
        if p == j or p < 0:
            roots.append(j)
        else:
            children[p].append(j)

    order = []
    queue = roots[:]
    while queue:
        j = queue.pop(0)
        order.append(j)
        queue.extend(children[j])
    return torch.tensor(order, device=parents.device, dtype=torch.long)


@dataclass
class Rig:
    """Canonical rig used by the solver."""

    X_rest: torch.Tensor  # (K,3) joint positions in rest space
    parents: torch.Tensor  # (K,) parent index, root has parent==itself
    W: torch.Tensor  # (Nv,K) skinning weights (rows sum to 1)
    valid: Optional[torch.Tensor] = None  # (K,) bool mask, optional

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "Rig":
        if dtype is None:
            dtype = self.X_rest.dtype
        return Rig(
            X_rest=self.X_rest.to(device=device, dtype=dtype),
            parents=self.parents.to(device=device),
            W=self.W.to(device=device, dtype=dtype),
            valid=None if self.valid is None else self.valid.to(device=device),
        )


class PoseFittingSolver:
    """Per-frame optimization of joint transforms to match a target mesh via LBS."""

    def __init__(
        self,
        rig: Rig,
        optimize_local_translation: bool = True,
        reg_rot: float = 1e-4,
        reg_trans: float = 1e-4,
    ) -> None:
        self.rig = rig
        self.optimize_local_translation = bool(optimize_local_translation)
        self.reg_rot = float(reg_rot)
        self.reg_trans = float(reg_trans)

        self._order = _topo_order(rig.parents)
        self._T_rest_global = self._compute_rest_globals(rig.X_rest, rig.parents, self._order)
        self._T_rest_global_inv = invert_transform(self._T_rest_global)  # (K,4,4)

        # Rest offsets in local space
        K = rig.parents.numel()
        offsets = torch.zeros((K, 3), device=rig.X_rest.device, dtype=rig.X_rest.dtype)
        for j in range(K):
            p = int(rig.parents[j])
            if p == j or p < 0:
                offsets[j] = rig.X_rest[j]
            else:
                offsets[j] = rig.X_rest[j] - rig.X_rest[p]
        self._rest_offsets = offsets

        if rig.valid is None:
            self._valid = torch.ones((K,), device=rig.X_rest.device, dtype=torch.bool)
        else:
            self._valid = rig.valid.bool()

    @staticmethod
    def _compute_rest_globals(X_rest: torch.Tensor, parents: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """Compute global transforms at rest pose (identity rotations)."""
        K = parents.numel()
        device, dtype = X_rest.device, X_rest.dtype
        T_global = torch.zeros((K, 4, 4), device=device, dtype=dtype)

        offsets = torch.zeros((K, 3), device=device, dtype=dtype)
        for j in range(K):
            p = int(parents[j])
            if p == j or p < 0:
                offsets[j] = X_rest[j]
            else:
                offsets[j] = X_rest[j] - X_rest[p]

        I = torch.eye(3, device=device, dtype=dtype).expand(K, 3, 3)
        T_local = make_transform(I, offsets)  # (K,4,4)

        for j in order.tolist():
            p = int(parents[j])
            if p == j or p < 0:
                T_global[j] = T_local[j]
            else:
                T_global[j] = T_global[p] @ T_local[j]
        return T_global

    def _forward_kinematics(
        self,
        aa_local: torch.Tensor,  # (K,3)
        dt_local: torch.Tensor,  # (K,3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute global transforms and joint world positions for current params."""
        R = axis_angle_to_matrix(aa_local)  # (K,3,3)
        t_local = self._rest_offsets + dt_local
        T_local = make_transform(R, t_local)  # (K,4,4)
        
        K = self.rig.parents.numel()
        # 用一个 Python 列表来暂存每个关节的 4x4 变换矩阵，避免原地修改 Tensor
        T_global_list = [None] * K
        
        for j in self._order.tolist():
            p = int(self.rig.parents[j])
            if p == j or p < 0:
                T_global_list[j] = T_local[j]
            else:
                T_global_list[j] = T_global_list[p] @ T_local[j]
                
        # 循环结束后，一次性沿着第 0 维度拼接成 (K, 4, 4) 的 Tensor
        T_global = torch.stack(T_global_list, dim=0)
        # joint world positions are transform of origin => translation column
        X_world = T_global[:, :3, 3]
        return T_global, X_world

    def lbs(
        self,
        V_rest: torch.Tensor,
        T_global: torch.Tensor,
        *,
        W: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply LBS using global transforms, returning posed vertices (Nv,3).

        Args:
            V_rest: (Nv,3)
            T_global: (K,4,4)
            W: optional (Nv,K) weights. If None, uses self.rig.W.
        """
        if W is None:
            W_use = self.rig.W
        else:
            W_use = W

        # Skinning transforms: G_k = T_k @ inv(T_rest_k)
        G = T_global @ self._T_rest_global_inv  # (K,4,4)

        Vh = torch.cat(
            [V_rest, torch.ones((V_rest.shape[0], 1), device=V_rest.device, dtype=V_rest.dtype)],
            dim=-1,
        )  # (Nv,4)

        # Apply each joint transform to all vertices: (K,Nv,3)
        V_k = (G[:, None, :, :] @ Vh[None, :, :, None]).squeeze(-1)[..., :3]

        # sanitize weights + valid mask
        W_use = W_use.to(device=V_rest.device, dtype=V_rest.dtype)
        if self._valid is not None:
            W_use = W_use.clone()
            W_use[:, ~self._valid] = 0.0
        W_use = W_use / (W_use.sum(dim=-1, keepdim=True).clamp_min(1e-8))

        # Weighted sum over joints: (Nv,3)
        V_hat = (W_use.t()[:, :, None] * V_k).sum(dim=0)
        return V_hat

    def fit_frame(
        self,
        V_rest: torch.Tensor,
        V_target: torch.Tensor,
        *,
        sample_verts: Optional[int] = 5000,
        num_iters: int = 80,
        lr: float = 1e-2,
        optimizer: Literal["adam", "lbfgs"] = "adam",
        verbose: bool = False,
        init_aa: Optional[torch.Tensor] = None,
        init_dt: Optional[torch.Tensor] = None,
        return_params: bool = False,
    ):
        """Fit a single frame.

        Args:
            V_rest: (Nv,3) rest vertices (anchor frame)
            V_target: (Nv,3) target vertices for this frame
            sample_verts: subsample count for optimization; None uses all vertices
            init_aa/init_dt: optional warm-start initial values (K,3)
            return_params: if True, also returns (aa, dt) for warm-start

        Returns:
            if return_params=False:
                (V_hat, X_world, T_global)
            else:
                (V_hat, X_world, T_global, aa, dt)
        """
        device = self.rig.X_rest.device
        dtype = self.rig.X_rest.dtype

        V_rest = V_rest.to(device=device, dtype=dtype)
        V_target = V_target.to(device=device, dtype=dtype)

        Nv = V_rest.shape[0]
        if sample_verts is not None and sample_verts < Nv:
            idx = torch.randperm(Nv, device=device)[: int(sample_verts)]
            V_rest_s = V_rest[idx]
            V_tgt_s = V_target[idx]
            W_s = self.rig.W[idx]
        else:
            idx = None
            V_rest_s, V_tgt_s = V_rest, V_target
            W_s = self.rig.W

        K = self.rig.parents.numel()

        if init_aa is None:
            aa0 = torch.zeros((K, 3), device=device, dtype=dtype)
        else:
            aa0 = init_aa.to(device=device, dtype=dtype).clone()
            if aa0.shape != (K, 3):
                raise ValueError(f"init_aa has shape {tuple(aa0.shape)}, expected {(K,3)}")

        if init_dt is None:
            dt0 = torch.zeros((K, 3), device=device, dtype=dtype)
        else:
            dt0 = init_dt.to(device=device, dtype=dtype).clone()
            if dt0.shape != (K, 3):
                raise ValueError(f"init_dt has shape {tuple(dt0.shape)}, expected {(K,3)}")

        aa = nn.Parameter(aa0)
        dt = nn.Parameter(dt0)
        if not self.optimize_local_translation:
            dt.requires_grad_(False)

        def loss_fn() -> torch.Tensor:
            T_global, _ = self._forward_kinematics(aa, dt)
            V_hat_s = self.lbs(V_rest_s, T_global, W=W_s)
            data = F.mse_loss(V_hat_s, V_tgt_s)
            reg = self.reg_rot * aa.pow(2).mean()
            if self.optimize_local_translation:
                reg = reg + self.reg_trans * dt.pow(2).mean()
            return data + reg

        if optimizer == "adam":
            opt = torch.optim.Adam([p for p in [aa, dt] if p.requires_grad], lr=lr)
            for it in range(int(num_iters)):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                opt.step()
                if verbose and (it % 10 == 0 or it == int(num_iters) - 1):
                    print(f"iter {it:03d} loss={loss.item():.6f}")
        elif optimizer == "lbfgs":
            opt = torch.optim.LBFGS(
                [p for p in [aa, dt] if p.requires_grad],
                max_iter=int(num_iters),
                line_search_fn="strong_wolfe",
            )

            def closure():
                opt.zero_grad(set_to_none=True)
                loss = loss_fn()
                loss.backward()
                return loss

            opt.step(closure)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Final forward on full vertices
        aa_f = aa.detach()
        dt_f = dt.detach()
        T_global, X_world = self._forward_kinematics(aa_f, dt_f)
        V_hat = self.lbs(V_rest, T_global, W=self.rig.W)

        if return_params:
            return V_hat, X_world, T_global, aa_f, dt_f
        return V_hat, X_world, T_global
