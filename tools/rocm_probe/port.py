"""Extract the three PTX-using e-means kernels, swap each PTX snippet for plain
Triton ops, and compile offline for AMD targets. Reads the e-means source as text only."""
import ast, re, sys
from collections import Counter
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

SRC = open(sys.argv[1]).read()
tree = ast.parse(SRC)
seg = lambda n: ast.get_source_segment(SRC, n)

def repl(call):
    asm = "".join(ast.literal_eval(call.args[0]).split())
    args = [seg(a) for a in next(k.value for k in call.keywords if k.arg == "args").elts]
    if asm == "fma.rn.f32$0,$1,$2,$3;":
        return "tl.fma(%s, %s, %s)" % tuple(args)
    if asm == "mul.rn.f32$0,$1,$2;":
        return "((%s) * (%s))" % tuple(args)
    if asm.startswith("{.reg.f32increment,total;"):
        return "_rn_step_mass(%s, %s, %s)" % tuple(args)
    if asm.startswith("{.reg.f32delta,scaled;"):
        return "_rn_lerp(%s, %s, %s)" % tuple(args)
    if asm.startswith("{.reg.f32offset,minus,plus,low,high;"):
        return "_rn_split(%s, %s, %s)" % tuple(args)
    if asm.startswith("cvt.rn.bf16.f32"):
        return "_rn_bf16_bits(%s, %s)" % tuple(args)
    if asm == "div.rn.f32$0,$1,0f40000000;":
        return "((%s) * 0.5)" % args[0]
    raise KeyError(asm)

def port(name):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    text = seg(fn)
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "inline_asm_elementwise"]
    for c in sorted(calls, key=lambda c: -len(seg(c))):
        assert seg(c) in text, name
        text = text.replace(seg(c), repl(c))
    assert "inline_asm" not in text
    deco = "".join("@" + seg(d) + "\n" for d in fn.decorator_list)
    return len(calls), deco + text

HELPERS = '''
@triton.jit
def _rn_step_mass(alpha, count, retained):
    step = alpha * count
    return step, retained + step

@triton.jit
def _rn_lerp(mean, centroid, weight):
    return (mean - centroid) * weight + centroid

@triton.jit
def _rn_split(value, epsilon, sign):
    offset = epsilon * sign
    return value * (1.0 - offset), value * (1.0 + offset)

@triton.jit
def _rn_bf16_bits(low, high):
    return (low.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.uint16, bitcast=True),
            high.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.uint16, bitcast=True))
'''
total = 0
body = "import triton\nimport triton.language as tl\n_fp8_libdevice = None\n" + HELPERS
for name in ("_cast_fp8_centroids_kernel", "_ema_update_kernel", "_write_float_splits_kernel"):
    n, text = port(name); total += n
    body += "\n\n" + text + "\n"
open("ported_kernels.py", "w").write(body)
import ported_kernels
ns = vars(ported_kernels)
print("PTX call sites replaced:", total)

F, I, P = "fp32", "i32", "*fp32"
CASES = {
 "ema fp32-sums DECAY": ("_ema_update_kernel", dict(sums=P, counts="*i64", centroids=P, usage=P, k=I, d=I, sum_stride="i64", count_stride="i64", alpha=F, decay=F), dict(DECAY=True, BLOCK_K=2, BLOCK_D=1024, SCALARS=False)),
 "ema replace-usage scalars": ("_ema_update_kernel", dict(sums=P, counts="*i64", centroids=P, usage=P, k=I, d=I, sum_stride="i64", count_stride="i64", alpha=P, decay=P), dict(DECAY=False, BLOCK_K=16, BLOCK_D=128, SCALARS=True)),
 "split fp32": ("_write_float_splits_kernel", dict(centroids=P, usage=P, source="*i64", target="*i64", d=I, epsilon=F, scale=F, decode_exponent="*i32"), dict(BF16=False, FP8=False, SCALED=False, BLOCK_D=128)),
 "split bf16": ("_write_float_splits_kernel", dict(centroids=P, usage=P, source="*i64", target="*i64", d=I, epsilon=F, scale=F, decode_exponent="*i32"), dict(BF16=True, FP8=False, SCALED=False, BLOCK_D=128)),
 "split fp8 unscaled": ("_write_float_splits_kernel", dict(centroids=P, usage=P, source="*i64", target="*i64", d=I, epsilon=F, scale=F, decode_exponent="*i32"), dict(BF16=False, FP8=True, SCALED=False, BLOCK_D=128)),
 "cast fp8 affine": ("_cast_fp8_centroids_kernel", dict(source=P, out="*fp8e4nv", squares=P, peak="*i32", bias=P, k=I, d=I, scale=F), dict(width=128, QUANTIZED=True, AFFINE=True, BLOCK=2048)),
 "cast fp8 scaled": ("_cast_fp8_centroids_kernel", dict(source=P, out="*fp8e4nv", squares=P, peak="*i32", bias=P, k=I, d=I, scale=F), dict(width=128, QUANTIZED=True, AFFINE=False, BLOCK=2048)),
}
FUSED = re.compile(r"^\s+(v_fma\w*|v_mad\w*|v_mac\w*)", re.M)
ARITH = re.compile(r"^\s+(v_(?:pk_)?(?:mul|add|sub)_f32\w*|v_div_\w+|v_rcp\w*|v_cvt_pk_bf16\w*)", re.M)
for arch in sys.argv[2:]:
    for label, (fn, sig, cx) in CASES.items():
        sig = dict(sig, **{k: "constexpr" for k in cx})
        out = {}
        for fusion in (False, True):
            try:
                k = triton.compile(ASTSource(fn=ns[fn], signature=sig, constexprs=cx),
                                   target=GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64),
                                   options=dict(num_warps=4, enable_fp_fusion=fusion))
                asm = k.asm["amdgcn"]
                out[fusion] = (len(k.asm.get("hsaco", b"")), dict(Counter(FUSED.findall(asm))), dict(Counter(ARITH.findall(asm))))
            except BaseException as e:
                out[fusion] = "FAILED %s: %s" % (type(e).__name__, str(e)[:160].replace("\n", " "))
        print("%-8s %-26s nofusion=%s" % (arch, label, out[False]))
        print("%-8s %-26s   fusion=%s" % ("", "", out[True] if isinstance(out[True], str) else out[True][1]))
