import re, sys
from collections import Counter
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import ported_kernels as pk
src = open("ported_kernels.py").read().split("\n")
F, I, P = "fp32", "i32", "*fp32"
sig = dict(sums=P, counts="*i64", centroids=P, usage=P, k=I, d=I, sum_stride="i64", count_stride="i64", alpha=P, decay=P)
cx = dict(DECAY=True, BLOCK_K=16, BLOCK_D=128, SCALARS=True)
sig.update({k: "constexpr" for k in cx})
for arch in sys.argv[1:]:
    for fusion in (False, True):
        k = triton.compile(ASTSource(fn=pk._ema_update_kernel, signature=sig, constexprs=cx),
                           target=GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64), options=dict(num_warps=4, enable_fp_fusion=fusion))
        line, hits = None, Counter()
        for l in k.asm["amdgcn"].split("\n"):
            m = re.match(r"\s+\.loc\s+\d+\s+(\d+)", l)
            if m: line = int(m.group(1)); continue
            if re.match(r"\s+v_(fma|fmac|mad_f|mac_f)\w*", l) and "mix" not in l:
                hits[line] += 1
        print(arch, "fusion=%s" % fusion)
        for ln, c in sorted(hits.items()):
            print("   %3d x  L%-3d %s" % (c, ln, src[ln - 1].strip()[:90]))
