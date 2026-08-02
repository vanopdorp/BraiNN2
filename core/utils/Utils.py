import torch
import subprocess
import modular
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).parent
MOJO_FILE = ROOT / "fast_ops.mojo"
MOJO_MODULE = ROOT / "fast_ops"


def _build_mojo_module():
    if not MOJO_MODULE.exists():
        print("Building Mojo module...")
        subprocess.run(
            ["mojo", "build", str(MOJO_FILE), "-o", str(MOJO_MODULE)],
            check=True
        )


_build_mojo_module()
fast_ops = modular.load(str(MOJO_MODULE))


def clamp_update(dw, max_norm):
    dw16 = dw.to(torch.float16).cpu().tolist()
    max16 = max_norm.to(torch.float16).cpu().tolist()

    out = fast_ops.clamp_update(dw16, max16)
    return torch.tensor(out, dtype=torch.float16, device=dw.device)


def soft_wta(x):
    x16 = x.to(torch.float16).cpu().tolist()

    out = fast_ops.soft_wta(x16)
    return torch.tensor(out, dtype=torch.float16, device=x.device)
