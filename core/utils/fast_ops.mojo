fn clamp_update(dw: List[f16], max_norm: List[f16]) -> List[f16]:
    let n = dw.size
    let out = List[f16](n)
    for i in range(n):
        let d = dw[i]
        let m = max_norm[i]
        out[i] = d * (m / (abs(d) + 1.0))
    return out


fn soft_wta(x: List[f16]) -> List[f16]:
    let n = x.size
    let relu = List[f16](n)
    var s: f32 = 1.0

    for i in range(n):
        let v = x[i]
        let r = max(v, 0.0)
        relu[i] = r
        s += r

    let out = List[f16](n)
    for i in range(n):
        out[i] = relu[i] / s

    return out
