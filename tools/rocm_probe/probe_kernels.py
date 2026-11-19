import triton
import triton.language as tl
@triton.jit
def plain(a_ptr, b_ptr, c_ptr, out_ptr, bf_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(a_ptr + i, mask=m, other=0.0)
    b = tl.load(b_ptr + i, mask=m, other=0.0)
    c = tl.load(c_ptr + i, mask=m, other=0.0)
    # same shape as e-means: (mean - centroid) * weight + centroid, then mul+add chain
    pos = (a - b) * c + b
    step = a * c
    mass = b + step
    half = tl.math.div_rn(mass, 2.0)
    w = tl.math.div_rn(step, mass)
    f = tl.fma(a, b, c)
    tl.store(out_ptr + i, pos + half + w + f, mask=m)
    tl.store(bf_ptr + i, pos.to(tl.bfloat16), mask=m)

@triton.jit
def ptx(a_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + i, mask=i < n, other=0.0)
    r = tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                  args=[a, a], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(out_ptr + i, r, mask=i < n)

@triton.jit
def gcn(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + i, mask=i < n, other=0.0)
    b = tl.load(b_ptr + i, mask=i < n, other=0.0)
    r = tl.inline_asm_elementwise("v_mul_f32 $0, $1, $2", constraints="=v,v,v",
                                  args=[a, b], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(out_ptr + i, r + b, mask=i < n)

