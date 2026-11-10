"""Differentiable, small-scale IsoKDN shrink experiment (no fused gate VJP).

Both gamma=1 control and gamma<1 treatment use this same FP32 recurrence.
Only the causal principal-axis estimate is detached; memory is never detached.
This deliberately favors a reviewable experiment over production throughput.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def validate_shrink(gamma, chunk, window, rank, key_dim):
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("shrink gamma must be in [0, 1]; 1 is the control")
    for name, value in (("chunk", chunk), ("window", window), ("rank", rank)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"shrink {name} must be a positive integer")
    if rank > key_dim:
        raise ValueError("shrink rank cannot exceed key dimension")


@torch.no_grad()
def past_key_basis(keys, rank):
    """Uncentered key moment, independently per example/head; no RNG consumed."""
    x = keys.transpose(1, 2)  # B,H,T,K
    moment = x.transpose(-1, -2) @ x / x.shape[-2]
    values, vectors = torch.linalg.eigh(moment)
    # Do not edit arbitrary eigenvectors of an unsupported/null subspace.
    supported = values[..., -rank:] > values[..., -1:].clamp_min(1e-12) * 1e-6
    return vectors[..., -rank:] * supported.unsqueeze(-2)


def shrink_recurrence(q, k, v, alpha, beta, *, gamma=1., chunk=16,
                      window=256, rank=1):
    """Normalized q/k, per-channel alpha, scalar beta; tensors B,T,H,D.

    Edit the decayed writes of a completed chunk after its last output. Use
    only keys preceding that chunk. Skip first and final chunks, matching the
    inference experiment. Older carry and scalar precision are not edited.
    """
    validate_shrink(gamma, chunk, window, rank, k.shape[-1])
    state = q.new_zeros(q.shape[0], q.shape[2], q.shape[-1], v.shape[-1])
    outputs = []
    length = q.shape[1]
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        editing = gamma < 1 and start > 0 and end < length
        basis = past_key_basis(k[:, max(0, start-window):start], rank) if editing else None
        update = torch.zeros_like(state) if editing else None
        for t in range(start, end):
            decay = alpha[:, t].unsqueeze(-1)
            state = state * decay
            residual = v[:, t] - (k[:, t].unsqueeze(-1) * state).sum(-2)
            increment = (beta[:, t, :, None] * k[:, t]).unsqueeze(-1) * residual.unsqueeze(-2)
            state = state + increment
            if editing:
                update = update * decay + increment
            outputs.append((q[:, t].unsqueeze(-1) * state).sum(-2) * q.shape[-1] ** -.5)
        if editing:
            state = state - (1-gamma) * (basis @ (basis.transpose(-1, -2) @ update))
    return torch.stack(outputs, dim=1)


def iso_shrink_forward(q, k, v, g_raw, omega_r_raw, omega_bias, r_bias,
                       A_log, dt_bias, initial_precision_param, *, omega_min,
                       r_min, info_scale, gamma, chunk, window, rank):
    """Ordinary autograd gates/gain; never consume the private paired prefix."""
    with torch.autocast(device_type=q.device.type, enabled=False):
        # Mirror the fused path's FP32 normalization then input-dtype rounding.
        def unit(x):
            y = x.float()
            return (y * torch.rsqrt(y.square().sum(-1, keepdim=True) + 1e-6)).to(x.dtype).float()

        q_unit, k_unit = unit(q), unit(k)
        heads, key_dim = k.shape[-2:]
        alpha = torch.exp(-A_log.float().reshape(1, 1, heads, 1).exp()
                         * F.softplus(g_raw.float() + dt_bias.float().reshape(heads, key_dim)))
        omega_raw, r_raw = omega_r_raw.float().split(heads, dim=-1)
        omega = omega_min + F.softplus(omega_raw + omega_bias.float())
        r = r_min + F.softplus(r_raw + r_bias.float())
        precision = .1 + F.softplus(initial_precision_param.float())
        a = alpha.square().mean(-1)
        gains = []
        for t in range(q.shape[1]):
            predicted = a[:, t] / precision + omega[:, t]
            # Production gain assumes normalized keys have unit squared norm.
            gains.append(predicted / (r[:, t] + predicted))
            precision = predicted.reciprocal() + info_scale / (key_dim * r[:, t])
        output = shrink_recurrence(q_unit, k_unit, v.float(), alpha,
                                   torch.stack(gains, dim=1), gamma=gamma,
                                   chunk=chunk, window=window, rank=rank)
    return output.to(v.dtype)
