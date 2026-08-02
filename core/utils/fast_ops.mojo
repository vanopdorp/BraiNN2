import torch
import modular

fast_ops = modular.load("core/utils/fast_ops.mojo")

def clamp_update(dw, max_norm):
    out = fast_ops.clamp_update(
        dw.to(torch.float16).cpu().tolist(),
        max_norm.to(torch.float16).cpu().tolist()
    )
    return torch.tensor(out, dtype=torch.float16, device=dw.device)

def soft_wta(x):
    out = fast_ops.soft_wta(
        x.to(torch.float16).cpu().tolist()
    )
    return torch.tensor(out, dtype=torch.float16, device=x.device)
