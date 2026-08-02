import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clamp_update(dw: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:

    if max_norm.numel() == 1:
        max_norm = max_norm.expand_as(dw)

    return dw * (max_norm / (torch.abs(dw) + 1.0))


def soft_wta(x: torch.Tensor) -> torch.Tensor:

    relu = torch.relu(x)
    s = relu.sum() + 1.0
    return relu / s

