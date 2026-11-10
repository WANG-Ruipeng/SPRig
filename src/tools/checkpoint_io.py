"""Strict, tensor-only loading shared by the release inference entry points."""
from collections.abc import Mapping


def load_strict(model, path):
    import torch
    pkg = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(pkg, Mapping):
        raise ValueError("Checkpoint must contain a tensor state dictionary.")
    state = pkg.get("model", pkg.get("state_dict", pkg))
    if not isinstance(state, Mapping) or not state or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
    ):
        raise ValueError("Checkpoint does not contain a tensor-only model state.")
    # Never silently ignore a wrong checkpoint, missing key or shape mismatch.
    model.load_state_dict(state, strict=True)
    del pkg, state
