from pathlib import Path
from torch.utils.cpp_extension import load

root = Path(__file__).parent

ops = load(
    name="ops",
    sources=[
        root / "ops_extension.cpp",
        root / "clamp_update.cu",
        root / "soft_wta.cu",
    ],
    verbose=True,
)


def clamp_update_cuda(dw, max_norm):
    return ops.clamp_update_fp16(dw)

def soft_wta_cuda(x):
    return ops.soft_wta_fp16(x)
