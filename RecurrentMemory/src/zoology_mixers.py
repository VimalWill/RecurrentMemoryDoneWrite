"""Adapters that let zoology's trainer run GDN-2, KDN and MesaNet.

zoology builds a mixer as ``Mixer(d_model=..., layer_idx=...)``, expects a plain
tensor from ``forward`` and reports size through ``state_size``. Every adapter
runs under BF16 autocast, which KDN requires for training and zoology's trainer
doesn't set up, so all mixers train at the same precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Adapter(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.mixer(hidden_states)[0]
        return output.float()

    def state_size(self, sequence_length: int = 0) -> int:
        """Recurrent floats in this layer. It doesn't depend on sequence length."""
        raise NotImplementedError


class GDN2Mixer(_Adapter):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, layer_idx: int | None = None, **kwargs):
        super().__init__()
        from LinearAttn.gdn2 import GatedDeltaNet2

        self.mixer = GatedDeltaNet2(
            hidden_size=d_model, num_heads=num_heads, head_dim=head_dim, layer_idx=layer_idx, **kwargs
        )
        self._memory = num_heads * head_dim * head_dim

    def state_size(self, sequence_length: int = 0) -> int:
        return self._memory


class IsoKDNMixer(_Adapter):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, layer_idx: int | None = None, **kwargs):
        super().__init__()
        from LinearAttn.iso_kdn import IsotropicKalmanDeltaNetwork

        self.mixer = IsotropicKalmanDeltaNetwork(
            hidden_size=d_model, num_heads=num_heads, head_dim=head_dim, layer_idx=layer_idx, **kwargs
        )
        self._size = num_heads * head_dim * head_dim + num_heads  # Plus one uncertainty per head.

    def state_size(self, sequence_length: int = 0) -> int:
        return self._size


class DiagKDNMixer(_Adapter):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, layer_idx: int | None = None, **kwargs):
        super().__init__()
        from LinearAttn.diag_kdn import DiagonalKalmanDeltaNetwork

        self.mixer = DiagonalKalmanDeltaNetwork(
            hidden_size=d_model, num_heads=num_heads, head_dim=head_dim, layer_idx=layer_idx, **kwargs
        )
        self._size = num_heads * head_dim * head_dim + num_heads * head_dim  # Plus uncertainty per channel.
        self._restore_omega_init = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Diagonal KDN starts from isotropic process noise, using a zero-initialized
        # projection. zoology re-initializes every Linear after building the model,
        # so restore it here, before the first optimizer step.
        if self._restore_omega_init:
            with torch.no_grad():
                for module in self.mixer.modules():
                    if getattr(module, "_kdn_omega_zero", False):
                        nn.init.zeros_(module.weight)
                        if module.bias is not None:
                            nn.init.zeros_(module.bias)
            self._restore_omega_init = False
        return super().forward(hidden_states)

    def state_size(self, sequence_length: int = 0) -> int:
        return self._size


class MesaNetMixer(_Adapter):
    def __init__(self, d_model: int, num_heads: int, head_dim: int, layer_idx: int | None = None, **kwargs):
        super().__init__()
        from LinearAttn.mesa_net import MesaNet

        self.mixer = MesaNet(
            hidden_size=d_model, num_heads=num_heads, head_dim=head_dim, layer_idx=layer_idx, **kwargs
        )
        self._memory = 2 * num_heads * head_dim * head_dim  # H_kk and H_kv.

    def state_size(self, sequence_length: int = 0) -> int:
        return self._memory
