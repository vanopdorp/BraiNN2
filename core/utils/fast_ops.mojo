import std.math

def clamp_update(dw: List[Float16], max_norm: List[Float16]) -> List[Float16]:
    var out: List[Float16] = List[Float16](length=len(dw), fill=0.0)
    for i in range(len(dw)):
        var d = dw[i]
        var m = max_norm[i]
        var a = std.math.abs(d)
        out[i] = d * (m / (a + 1.0))
    return out^


def soft_wta(x: List[Float16]) -> List[Float16]:
    var relu: List[Float16] = List[Float16](length=len(x), fill=0.0)
    var s: Float32 = 1.0

    for i in range(len(x)):
        var v = x[i]
        var r = std.math.max(v, 0.0)
        relu[i] = r
        s += Float32(r)

    var out: List[Float16] = List[Float16](length=len(x), fill=0.0)
    for i in range(len(x)):
        out[i] = relu[i] / Float16(s)
    return out^
