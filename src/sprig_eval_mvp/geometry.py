"""Minimal geometric utilities for SPRig evaluation MVP.

This module intentionally stays small and dependency-light.
"""

from __future__ import annotations

import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert axis-angle vectors (..., 3) to rotation matrices (..., 3, 3).

    Uses Rodrigues' formula with a small-angle stable branch.

    Args:
        axis_angle: Tensor of shape (..., 3). Vector direction is axis, length is angle (radians).
        eps: Small constant for numerical stability.

    Returns:
        Rotation matrices of shape (..., 3, 3).
    """
    aa = axis_angle
    orig_shape = aa.shape
    assert orig_shape[-1] == 3, f"axis_angle must end with 3, got {orig_shape}"
    aa = aa.reshape(-1, 3)

    theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(eps)  # (N,1)
    axis = aa / theta  # (N,3)
    x, y, z = axis.unbind(dim=-1)

    # Skew-symmetric matrix K for each axis
    zeros = torch.zeros_like(x)
    K = torch.stack(
        [
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    # Rodrigues: R = I + sinθ K + (1-cosθ) K^2
    I = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0).expand(aa.shape[0], 3, 3)
    sin_t = torch.sin(theta).reshape(-1, 1, 1)
    cos_t = torch.cos(theta).reshape(-1, 1, 1)

    K2 = K @ K
    R = I + sin_t * K + (1.0 - cos_t) * K2

    # Small-angle correction using Taylor expansions around 0.
    # When theta ~ 0, axis is unstable; use first-order approximation: R ~ I + K*theta
    small = (theta.squeeze(-1) < 1e-4)
    if small.any():
        aa_small = aa[small]  # (Ns,3)
        # Build skew from raw aa (already includes angle)
        x2, y2, z2 = aa_small.unbind(dim=-1)
        Kaa = torch.stack(
            [
                torch.zeros_like(x2), -z2, y2,
                z2, torch.zeros_like(x2), -x2,
                -y2, x2, torch.zeros_like(x2)
            ],
            dim=-1,
        ).reshape(-1, 3, 3)
        I2 = torch.eye(3, device=aa.device, dtype=aa.dtype).unsqueeze(0).expand(Kaa.shape[0], 3, 3)
        R_small = I2 + Kaa  # first order
        R[small] = R_small

    return R.reshape(*orig_shape[:-1], 3, 3)


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Build homogeneous transform (..., 4, 4) from R (...,3,3) and t (...,3)."""
    assert R.shape[-2:] == (3, 3)
    assert t.shape[-1] == 3
    *batch, _, _ = R.shape
    T = torch.zeros((*batch, 4, 4), device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def invert_transform(T: torch.Tensor) -> torch.Tensor:
    """Invert a rigid transform (...,4,4). Assumes last row is [0,0,0,1]."""
    assert T.shape[-2:] == (4, 4)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    tinv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    return make_transform(Rt, tinv)


def transform_points(T: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """Apply transform T (...,4,4) to points V (...,N,3) -> (...,N,3)."""
    assert T.shape[-2:] == (4, 4)
    assert V.shape[-1] == 3
    ones = torch.ones((*V.shape[:-1], 1), device=V.device, dtype=V.dtype)
    Vh = torch.cat([V, ones], dim=-1)  # (...,N,4)
    out = (T @ Vh.unsqueeze(-1)).squeeze(-1)[..., :3]
    return out
