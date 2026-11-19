import re, sys, inspect
import triton, triton.language as tl
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

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

def build(fn, sig, arch, **opts):
    src = ASTSource(fn=fn, signature=sig, constexprs={"BLOCK": 256})
    return triton.compile(src, target=GPUTarget("hip", arch, 64), options=opts)

FP = re.compile(r"\b(v_fma\w*|v_mad\w*|v_mac\w*|v_mul_f32|v_add_f32|v_sub_f32|v_div\w*|v_rcp\w*|v_ldexp\w*|v_cvt\w*)\b")
def ops(asm):
    from collections import Counter
    return dict(Counter(FP.findall(asm)))

sig3 = {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32", "out_ptr": "*fp32", "bf_ptr": "*bf16", "n": "i32", "BLOCK": "constexpr"}
for arch in sys.argv[1:]:
    for fusion in (False, True):
        try:
            k = build(plain, sig3, arch, enable_fp_fusion=fusion)
            print(arch, "plain fusion=%s" % fusion, ops(k.asm["amdgcn"]))
            if not fusion:
                open(f"plain_{arch}.s", "w").write(k.asm["amdgcn"]); open(f"plain_{arch}.ll", "w").write(k.asm["llir"])
        except Exception as e:
            print(arch, "plain fusion=%s FAILED" % fusion, type(e).__name__, str(e)[:300])
    for name, fn, sig in (("ptx", ptx, {"a_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"}),
                          ("gcn", gcn, {"a_ptr": "*fp32", "b_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"})):
        try:
            k = build(fn, sig, arch, enable_fp_fusion=True)
            print(arch, name, "inline asm OK", ops(k.asm["amdgcn"]))
        except BaseException as e:
            print(arch, name, "inline asm FAILED", type(e).__name__, str(e)[:300].replace("\n", " | "))
