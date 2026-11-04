"""Linear attention layers.

Layers load on first use, so importing this package doesn't require CUDA or fla.
"""
from importlib import import_module

_LAYERS = {
    "GatedDeltaNet2": ".gdn2",
    "IsotropicKalmanDeltaNetwork": ".iso_kdn",
    "DiagonalKalmanDeltaNetwork": ".diag_kdn",
    "MesaNet": ".mesa_net",
}

__all__ = list(_LAYERS)


def __getattr__(name):
    if name in _LAYERS:
        return getattr(import_module(_LAYERS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
