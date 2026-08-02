import torch

# Zorg dat we op CUDA draaien als het kan
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.compile(mode="reduce-overhead")
def _clamp_update_compiled(dw: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:
    # Aanname: dw en max_norm zijn al fp16 en op hetzelfde device
    if max_norm.numel() == 1:
        max_norm = max_norm.expand_as(dw)
    return dw * (max_norm / (torch.abs(dw) + 1.0))


@torch.compile(mode="reduce-overhead")
def _soft_wta_compiled(x: torch.Tensor) -> torch.Tensor:
    # Aanname: x is al fp16 en op hetzelfde device
    relu = torch.relu(x)
    s = relu.sum() + 1.0
    return relu / s


def clamp_update(dw: torch.Tensor, max_norm: torch.Tensor) -> torch.Tensor:
    """
    Snelle fp16 clamp_update, gecompileerd met torch.compile.
    """
    dw = dw.to(device=DEVICE, dtype=torch.float16)
    max_norm = max_norm.to(device=DEVICE, dtype=torch.float16)
    return _clamp_update_compiled(dw, max_norm)


def soft_wta(x: torch.Tensor) -> torch.Tensor:
    """
    Snelle fp16 soft_wta, gecompileerd met torch.compile.
    """
    x = x.to(device=DEVICE, dtype=torch.float16)
    return _soft_wta_compiled(x)


if __name__ == "__main__":
    # Kleine sanity‑benchmark
    x = torch.randn(4096, device=DEVICE, dtype=torch.float16)
    max_norm = torch.full((4096,), 18.0, device=DEVICE, dtype=torch.float16)

    # Warmup (belangrijk voor torch.compile)
    for _ in range(10):
        clamp_update(x, max_norm)
        soft_wta(x)

    import timeit

    runs = 10000

    t_clamp = timeit.timeit(
        stmt="clamp_update(x, max_norm)",
        number=runs,
        globals={"clamp_update": clamp_update, "x": x, "max_norm": max_norm},
    )

    t_wta = timeit.timeit(
        stmt="soft_wta(x)",
        number=runs,
        globals={"soft_wta": soft_wta, "x": x},
    )

    clamp_us = (t_clamp / runs) * 1e6
    wta_us = (t_wta / runs) * 1e6

    print(f"Benchmark running on: {DEVICE}")
    print(f"clamp_update: {t_clamp:.4f} sec for {runs} runs ({clamp_us:.2f} µs per call)")
    print(f"soft_wta:     {t_wta:.4f} sec for {runs} runs ({wta_us:.2f} µs per call)")
