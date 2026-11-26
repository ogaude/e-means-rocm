import re
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import ported_kernels as pk, probe_kernels as pr
F, I, P = "fp32", "i32", "*fp32"
sig = dict(sums=P, counts="*i64", centroids=P, usage=P, k=I, d=I, sum_stride="i64", count_stride="i64", alpha=F, decay=F)
cx = dict(DECAY=True, BLOCK_K=2, BLOCK_D=1024, SCALARS=False)
sig.update({k: "constexpr" for k in cx})
for fusion in (False, True):
    try:
        k = triton.compile(ASTSource(fn=pk._ema_update_kernel, signature=sig, constexprs=cx),
                           target=GPUTarget("cuda", 90, 32), options=dict(num_warps=4, enable_fp_fusion=fusion))
        ptx = k.asm["ptx"]
    except BaseException as e:
        print("cuda compile fusion=%s: %s %s" % (fusion, type(e).__name__, str(e)[:200])); continue
    from collections import Counter
    print("cuda sm90 fusion=%s" % fusion, dict(Counter(re.findall(r"^\s+((?:mul|add|sub|div|fma)\.[a-z.]*f32)", ptx, re.M))))
# PTX snippet on an AMD target
s = {"a_ptr": P, "out_ptr": P, "n": I, "BLOCK": "constexpr"}
try:
    k = triton.compile(ASTSource(fn=pr.ptx, signature=s, constexprs={"BLOCK": 256}), target=GPUTarget("hip", "gfx942", 64))
    print("PTX snippet on hip: hsaco bytes =", len(k.asm.get("hsaco") or b""), "| amdgcn has v_mul:", "v_mul" in k.asm.get("amdgcn", ""))
except BaseException as e:
    print("PTX snippet on hip FAILED:", type(e).__name__, str(e)[:200])
