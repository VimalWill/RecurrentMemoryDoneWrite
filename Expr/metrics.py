"""Matrix measurements for state_geometry.py. Inputs are batches of matrices [..., m, n].

Singular values are divided by the largest before any ratio, so every measure is
unchanged by scaling the matrix. A zero matrix has effective, stable and
numerical rank 0.
"""
from __future__ import annotations

import math

import torch

# Singular values below this fraction of the largest count as zero.
RANK_TOLERANCE = 1e-3


def _relative(singular_values: torch.Tensor) -> torch.Tensor:
    """Singular values (sorted, largest first) divided by the largest; all zero for a zero matrix."""
    top = singular_values[..., :1]
    safe_top = torch.where(top > 0, top, torch.ones_like(top))
    return torch.where(top > 0, singular_values / safe_top, torch.zeros_like(singular_values))


def effective_rank(singular_values: torch.Tensor) -> torch.Tensor:
    """exp(entropy) of the normalized singular values (Roy and Vetterli, 2007).

    It runs from 1, when one direction holds everything, to the matrix size, when
    every direction is equally strong.
    """
    relative = _relative(singular_values)
    total = relative.sum(-1, keepdim=True)
    p = relative / torch.where(total > 0, total, torch.ones_like(total))
    erank = torch.exp(-torch.special.xlogy(p, p).sum(-1))
    return torch.where(total.squeeze(-1) > 0, erank, torch.zeros_like(erank))


def stable_rank(singular_values: torch.Tensor) -> torch.Tensor:
    """||M||_F^2 / ||M||_2^2: how many directions carry energy comparable to the largest."""
    return _relative(singular_values).pow(2).sum(-1)


def numerical_rank(singular_values: torch.Tensor) -> torch.Tensor:
    return (_relative(singular_values) > RANK_TOLERANCE).sum(-1).to(singular_values.dtype)


def polar(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return U, P and P's eigenvalues for M = U P, U orthogonal and P symmetric PSD."""
    left, singular_values, right_h = torch.linalg.svd(matrix)
    rotation = left @ right_h
    stretch = right_h.mT @ (singular_values[..., None] * right_h)
    return rotation, stretch, singular_values


def polar_summary(matrix: torch.Tensor) -> dict[str, torch.Tensor]:
    """Summarize a square matrix as rotation times stretch.

    full_rank       1 if every direction is above RANK_TOLERANCE. U is unique only
                    then: in directions M sends to zero, the SVD fills U with an
                    arbitrary rotation, so rotation_deg and reflection are NaN.
    rotation_deg    mean |angle| of U's eigenvalues; 0 means M is a pure stretch
    reflection      1 if U flips orientation (det U < 0)
    stretch_spread  std / mean of P's eigenvalues; 0 means equal stretch in every
                    direction, so M is a rotation times a scalar
    """
    rotation, _, singular_values = polar(matrix)
    relative = _relative(singular_values)
    rank = numerical_rank(singular_values)
    full_rank = rank == matrix.shape[-1]
    angles = torch.linalg.eigvals(rotation).angle().abs().mean(-1) * (180.0 / math.pi)
    reflection = (torch.linalg.det(rotation) < 0).to(relative.dtype)
    undefined = torch.full_like(angles, math.nan)
    mean = relative.mean(-1)
    return {
        "erank": effective_rank(singular_values),
        "stable_rank": stable_rank(singular_values),
        "numerical_rank": rank,
        "full_rank": full_rank.to(relative.dtype),
        "rotation_deg": torch.where(full_rank, angles, undefined),
        "reflection": torch.where(full_rank, reflection, undefined),
        "stretch_spread": torch.where(mean > 0, relative.std(-1) / torch.where(mean > 0, mean, 1.0), undefined),
    }
