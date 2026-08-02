import torch

def clamp_update(dw, max_norm):
    dw = dw.to(torch.float16)
    max_norm = max_norm.to(torch.float16)

    if max_norm.numel() == 1:
        max_norm = max_norm.expand_as(dw)

    return dw * (max_norm / (torch.abs(dw) + 1.0))


def soft_wta(x):
    x = x.to(torch.float16)

    relu = torch.relu(x)
    s = relu.sum() + 1.0

    return relu / s
