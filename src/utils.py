"""Small helpers shared by the local training and inference commands."""

import random
from pathlib import Path

import numpy as np
import torch


def seed_all(seed, deterministic=False):
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def resolve_device(requested="auto"):
    """Resolve ``auto``, ``cpu``, or ``cuda`` to a usable torch device."""
    requested = str(requested).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but this PyTorch install has no CUDA device.")
    return torch.device(requested)


def load_checkpoint(path, device="cpu"):
    """Load a project checkpoint and reject files without model weights."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Not a Visual Similarity Search checkpoint: {path}")
    return checkpoint
