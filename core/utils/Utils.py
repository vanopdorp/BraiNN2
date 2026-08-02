import torch
import triton
import triton.language as tl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def clamp_kernel(dw_ptr, max_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    dw = tl.load(dw_ptr + offsets, mask=mask)
    mx = tl.load(max_ptr + offsets, mask=mask)

    out = dw * (mx / (tl.abs(dw) + 1.0))
    tl.store(out_ptr + offsets, out, mask=mask)


def clamp_update(dw, max_norm):
    dw = dw.to(torch.float16).to(DEVICE)
    max_norm = max_norm.to(torch.float16).to(DEVICE)

    if max_norm.numel() == 1:
        max_norm = max_norm.expand_as(dw)

    out = torch.empty_like(dw)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(dw.numel(), BLOCK),)

    clamp_kernel[grid](dw, max_norm, out, dw.numel(), BLOCK=BLOCK)
    return out


@triton.jit
def soft_wta_kernel(x_ptr, out_ptr, sum_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    relu = tl.maximum(x, 0.0)

    tl.atomic_add(sum_ptr, tl.sum(relu, axis=0))

    tl.store(out_ptr + offsets, relu, mask=mask)


def soft_wta(x):
    x = x.to(torch.float16).to(DEVICE)
    out = torch.empty_like(x)
    sum_buf = torch.zeros(1, dtype=torch.float32, device=DEVICE)

    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(x.numel(), BLOCK),)

    soft_wta_kernel[grid](x, out, sum_buf, x.numel(), BLOCK=BLOCK)

    s = sum_buf.item() + 1.0
    return out / s
