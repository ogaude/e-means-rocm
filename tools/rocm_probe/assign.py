import ast, re, sys, warnings
warnings.filterwarnings("ignore")
SRC = open(sys.argv[1]).read(); tree = ast.parse(SRC)
names = ["_selected_fp8_distance", "_assign_kernel", "_assign_split_k_kernel", "_reduce_split_k_kernel",
         "_stage_int8_rows_kernel", "_prepare_int8_kernel", "_assign_int8_kernel"]
body = "import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice as _int8_libdevice\n"
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in names:
        body += "\n\n" + "".join("@" + ast.get_source_segment(SRC, d) + "\n" for d in n.decorator_list) + ast.get_source_segment(SRC, n) + "\n"
open("assign_kernels.py", "w").write(body)
import triton, assign_kernels as ak
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
def sig_assign(t):
    return dict(x_ptr="*"+t, centroid_ptr="*"+t, centroid_norm_ptr="*fp32", distance_ptr="*fp32", label_ptr="*i64",
                B="i32", K="i32", D="i32", stride_xb="i64", stride_xd="i64", stride_ck="i64", stride_cd="i64")
def A(t, bb, bk, bd, full, **kw):
    cx = dict(BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=bd, FULL_D=full, REDUCED=t != "fp32", FP16=t == "fp16", FP8=t == "fp8e4nv",
              WIDE_CENTROID=False, x_desc=None, centroid_desc=None, SPHERICAL=False)
    return ("_assign_kernel", sig_assign(t), cx, kw)
CASES = {
 "fp32 direct   w8 s2": A("fp32", 64, 128, 32, False, num_warps=8, num_stages=2),
 "bf16 direct   w8 s2": A("bf16", 64, 256, 64, False, num_warps=8, num_stages=2),
 "bf16 direct   w8 s3": A("bf16", 64, 256, 64, False, num_warps=8, num_stages=3),
 "bf16 b128     w8 s2": A("bf16", 128, 256, 64, False, num_warps=8, num_stages=2),
 "fp16 direct   w8 s2": A("fp16", 64, 256, 64, False, num_warps=8, num_stages=2),
 "bf16 full-d   w4 s2": A("bf16", 64, 128, 128, True, num_warps=4, num_stages=2),
 "fp8  direct   w8 s2": A("fp8e4nv", 64, 256, 64, False, num_warps=8, num_stages=2),
 "int8 direct   w8 s3": ("_assign_int8_kernel", dict(sig_assign("i8"), centroid_norm_ptr="*i32", distance_ptr="*fp32"),
                         dict(UNSIGNED=False, BLOCK_B=128, BLOCK_K=256, BLOCK_D=128, WIDE_CENTROID=False, FULL_D=False, PACKED=True, x_desc=None, centroid_desc=None),
                         dict(num_warps=8, num_stages=3)),
 "int8 direct   w8 s2": ("_assign_int8_kernel", dict(sig_assign("i8"), centroid_norm_ptr="*i32", distance_ptr="*fp32"),
                         dict(UNSIGNED=False, BLOCK_B=128, BLOCK_K=256, BLOCK_D=128, WIDE_CENTROID=False, FULL_D=False, PACKED=True, x_desc=None, centroid_desc=None),
                         dict(num_warps=8, num_stages=2)),
}
MM = re.compile(r"^\s+(v_mfma_\w+|v_wmma_\w+|v_swmmac\w*)", re.M)
for arch in sys.argv[2:]:
    for label, (fn, sig, cx, opts) in CASES.items():
        sig2 = dict(sig, **{k: "constexpr" for k in cx})
        try:
            k = triton.compile(ASTSource(fn=getattr(ak, fn), signature=sig2, constexprs=cx),
                               target=GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64), options=dict(enable_fp_fusion=False, **opts))
            mm = sorted(set(MM.findall(k.asm["amdgcn"])))
            shared = getattr(k.metadata, "shared", None)
            print("%-8s %-20s OK  lds=%-6s matrix=%s" % (arch, label, shared, ",".join(mm) or "none (scalar dot)"))
        except BaseException as e:
            print("%-8s %-20s FAIL %s: %s" % (arch, label, type(e).__name__, re.sub(r"\s+", " ", str(e))[:170]))
