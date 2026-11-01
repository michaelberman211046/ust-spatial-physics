"""CPU checkpoint loading that avoids allocating optimizer tensors on CUDA."""
import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, TypeError) as exc:
        if 'mmap' not in str(exc):
            raise
        return torch.load(path, map_location='cpu', weights_only=False)
