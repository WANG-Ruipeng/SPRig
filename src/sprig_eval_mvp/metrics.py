"""Evaluation metrics for SPRig MVP (new protocol).

Implements:
- LBS reconstruction RMSE / approximate Chamfer distance
- Temporal jitter (joint acceleration energy)
- Bone length variance (BLV)

These metrics are model-agnostic and operate on fitted joint trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

try:
    from pytorch3d.loss import chamfer_distance as _p3d_chamfer_distance  # type: ignore
except Exception:
    _p3d_chamfer_distance = None


def rmse(V_hat: torch.Tensor, V_gt: torch.Tensor) -> torch.Tensor:
    """Vertex-wise RMSE."""
    return torch.sqrt(torch.mean(torch.sum((V_hat - V_gt) ** 2, dim=-1)))


def chamfer_distance(
    V_hat: torch.Tensor,
    V_gt: torch.Tensor,
    *,
    num_samples: int = 5000,
) -> torch.Tensor:
    """Approximate symmetric Chamfer distance.

    If PyTorch3D is available, uses its chamfer_distance. Otherwise computes
    pairwise distances with subsampling (O(n^2) on samples).
    """
    device = V_hat.device
    n1 = V_hat.shape[0]
    n2 = V_gt.shape[0]
    if num_samples is not None:
        if n1 > num_samples:
            V_hat = V_hat[torch.randperm(n1, device=device)[:num_samples]]
        if n2 > num_samples:
            V_gt = V_gt[torch.randperm(n2, device=device)[:num_samples]]

    if _p3d_chamfer_distance is not None:
        # pytorch3d expects (B,N,3)
        d, _ = _p3d_chamfer_distance(V_hat[None], V_gt[None])
        return d.squeeze(0)

    # fallback: brute force on samples
    dists = torch.cdist(V_hat, V_gt)  # (N1,N2)
    cd = dists.min(dim=1).values.mean() + dists.min(dim=0).values.mean()
    return cd


def temporal_jitter_acceleration(X: torch.Tensor) -> torch.Tensor:
    """Joint acceleration energy.

    Args:
        X: (T,K,3) joint world coordinates.

    Returns:
        Scalar jitter = mean over t=1..T-2 and joints of ||x_{t-1}-2x_t+x_{t+1}||^2.
    """
    assert X.ndim == 3 and X.shape[-1] == 3
    if X.shape[0] < 3:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    acc = X[:-2] - 2.0 * X[1:-1] + X[2:]  # (T-2,K,3)
    return (acc.pow(2).sum(dim=-1)).mean()


def bone_length_variance(X: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Average per-bone length std over time.

    Args:
        X: (T,K,3) joint world coords
        parents: (K,) parent indices; root has parent==itself (or <0).

    Returns:
        Scalar BLV.
    """
    T, K, _ = X.shape
    bone_stds = []
    for j in range(K):
        p = int(parents[j])
        if p == j or p < 0:
            continue
        lengths = torch.linalg.norm(X[:, j] - X[:, p], dim=-1)  # (T,)
        bone_stds.append(lengths.std(unbiased=False))
    if len(bone_stds) == 0:
        return torch.zeros((), device=X.device, dtype=X.dtype)
    return torch.stack(bone_stds).mean()


@dataclass
class SequenceMetrics:
    rmse_mean: float
    chamfer_mean: float
    jitter: float
    blv: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "rmse": self.rmse_mean,
            "chamfer": self.chamfer_mean,
            "jitter": self.jitter,
            "blv": self.blv,
        }


def aggregate_metrics(
    V_hats: torch.Tensor,   # (T,Nv,3)
    V_gts: torch.Tensor,    # (T,Nv,3)
    X_world: torch.Tensor,  # (T,K,3)
    parents: torch.Tensor,  # (K,)
    *,
    chamfer_samples: int = 5000,
) -> SequenceMetrics:
    """Compute all metrics for a fitted sequence."""
    assert V_hats.shape == V_gts.shape and V_hats.ndim == 3
    T = V_hats.shape[0]
    rmses = []
    chs = []
    for t in range(T):
        rmses.append(rmse(V_hats[t], V_gts[t]))
        chs.append(chamfer_distance(V_hats[t], V_gts[t], num_samples=chamfer_samples))
    rmse_mean = torch.stack(rmses).mean().item()
    chamfer_mean = torch.stack(chs).mean().item()
    jitter = temporal_jitter_acceleration(X_world).item()
    blv = bone_length_variance(X_world, parents).item()
    return SequenceMetrics(rmse_mean=rmse_mean, chamfer_mean=chamfer_mean, jitter=jitter, blv=blv)
