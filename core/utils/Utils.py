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
    return ops.clamp_update_fp16(dw, max_norm)

def soft_wta(x):
    return ops.soft_wta_fp16(x)
