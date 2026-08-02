# core/utils/Utils.py
import torch
from pathlib import Path
from torch.utils.cpp_extension import load

root = Path(__file__).parent

ops = load(
    name="ops",
    sources=[
        root / "ops_extension.cpp",
        root / "ops_extension.cu",
        root / "clamp_update.cu",
        root / "soft_wta.cu",
    ],
    verbose=True,
)


def clamp_update(dw, max_norm):
    dw = dw.to(torch.float16)
    max_norm = max_norm.to(torch.float16)
    return ops.clamp_update_fp16(dw, max_norm)


def soft_wta(x):
    x = x.to(torch.float16)
    return ops.soft_wta_fp16(x)
