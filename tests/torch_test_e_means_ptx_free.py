# Offline compile checks for the Triton helper kernels that used to contain PTX inline
# assembly. No GPU is needed: kernels are compiled for AMD (and one NVIDIA) targets and
# the emitted assembly is inspected.
#
# What this guards:
#   * no inline assembly is left in triton_assignment.py (PTX on an AMD target compiles
#     "successfully" with the operation silently missing);
#   * with enable_fp_fusion=False, FMA-family instructions in the EMA kernel come only
#     from the two tl.math.div_rn lines, i.e. no multiply/add pair was contracted;
#   * the split kernels contain no FMA-family instruction at all;
#   * the FP8 cast kernel compiles for every target.

import re
import unittest
from collections import Counter

try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from faiss.contrib.e_means import triton_assignment as ta
    HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    HAVE_TRITON = False

AMD_ARCHS = ("gfx942", "gfx950", "gfx1100", "gfx1200")
# Scalar and packed FMA-family ops (v_pk_fma_f32 exists on gfx942/gfx950). Integer
# v_mad_u64_u32 and mixed-precision v_fma_mix* are excluded.
FUSED = re.compile(r"\s+v_(?:pk_)?(?:fma|fmac|mad_f|mac_f)\w*")
F32, I32, P = "fp32", "i32", "*fp32"


def _wave(arch):
    return 32 if arch.startswith("gfx1") else 64


def _compile(fn, signature, constexprs, arch, fusion=False):
    signature = dict(signature, **{name: "constexpr" for name in constexprs})
    return triton.compile(
        ASTSource(fn=fn, signature=signature, constexprs=constexprs),
        target=GPUTarget("hip", arch, _wave(arch)),
        options=dict(num_warps=4, enable_fp_fusion=fusion))


def _fused_lines(asm):
    """Source line numbers (in triton_assignment.py) of every FMA-family instruction.

    An instruction attributed to any other source file is reported as line -1, which
    fails the div_rn check instead of being looked up in the wrong file.
    """
    files = {int(index): name for index, name in
             re.findall(r'\.file\s+(\d+)\s+"[^"]*"\s+"([^"]+)"', asm)}
    files.update({int(index): name for index, name in
                  re.findall(r'\.file\s+(\d+)\s+"([^"]+)"\s*$', asm, re.M)})
    line, hits = None, Counter()
    for text in asm.split("\n"):
        loc = re.match(r"\s+\.loc\s+(\d+)\s+(\d+)", text)
        if loc:
            ours = files.get(int(loc.group(1)), "").endswith("triton_assignment.py")
            line = int(loc.group(2)) if ours else -1
        elif FUSED.match(text) and "mix" not in text:
            hits[line] += 1
    return hits


def _on_div_rn(lines, line):
    return line is not None and line > 0 and "div_rn" in lines[line - 1]


EMA_SIG = dict(sums=P, counts="*i64", centroids=P, usage=P, k=I32, d=I32,
               sum_stride="i64", count_stride="i64", alpha=P, decay=P)
EMA_CASES = {
    "decay, device scalars": dict(DECAY=True, BLOCK_K=16, BLOCK_D=128, SCALARS=True),
    "replace usage, device scalars": dict(DECAY=False, BLOCK_K=16, BLOCK_D=128, SCALARS=True),
}
EMA_SIG_HOST = dict(EMA_SIG, alpha=F32, decay=F32)
SPLIT_SIG = dict(centroids=P, usage=P, source="*i64", target="*i64", d=I32, epsilon=F32,
                 scale=F32, decode_exponent="*i32")
SPLIT_CASES = {
    "fp32": dict(BF16=False, FP8=False, SCALED=False, BLOCK_D=128, LIBDEVICE_LDEXP=True),
    "bf16": dict(BF16=True, FP8=False, SCALED=False, BLOCK_D=128, LIBDEVICE_LDEXP=True),
    "fp8 unscaled": dict(BF16=False, FP8=True, SCALED=False, BLOCK_D=128, LIBDEVICE_LDEXP=True),
    # AMD: ldexp replaced by an exact power-of-two multiply (no libdevice ldexp there).
    "fp8 scaled": dict(BF16=False, FP8=True, SCALED=True, BLOCK_D=128, LIBDEVICE_LDEXP=False),
}
CAST_SIG = dict(source=P, out="*fp8e4nv", squares=P, peak="*i32", bias=P, k=I32, d=I32, scale=F32)
CAST_CASES = {
    "affine": dict(width=128, QUANTIZED=True, AFFINE=True, BLOCK=2048),
    "scaled": dict(width=128, QUANTIZED=True, AFFINE=False, BLOCK=2048),
}


@unittest.skipUnless(HAVE_TRITON, "triton is not available")
class PtxFreeKernelsTest(unittest.TestCase):

    def test_no_inline_assembly_left(self):
        with open(ta.__file__) as handle:
            source = handle.read()
        self.assertNotIn("inline_asm_elementwise", source)

    def test_fusion_stays_disabled_on_launches(self):
        with open(ta.__file__) as handle:
            source = handle.read()
        for kernel in ("_cast_fp8_centroids_kernel", "_ema_update_kernel",
                       "_write_float_splits_kernel"):
            launch = source.split("%s[" % kernel, 1)[1].split(")\n", 1)[0]
            self.assertIn("enable_fp_fusion=False", launch, kernel)

    def test_ema_fma_only_on_div_rn_lines(self):
        with open(ta.__file__) as handle:
            lines = handle.read().split("\n")
        for arch in AMD_ARCHS:
            for label, constexprs in EMA_CASES.items():
                with self.subTest(arch=arch, case=label):
                    kernel = _compile(ta._ema_update_kernel, EMA_SIG, constexprs, arch)
                    self.assertGreater(len(kernel.asm.get("hsaco", b"")), 0)
                    hits = _fused_lines(kernel.asm["amdgcn"])
                    self.assertTrue(hits, "expected FMA-family instructions from div_rn")
                    for line, count in hits.items():
                        self.assertTrue(_on_div_rn(lines, line),
                                        "%d fused op(s) on line %d: %s"
                                        % (count, line, lines[line - 1].strip() if line > 0
                                           else "<outside triton_assignment.py>"))

    def test_ema_fusion_flag_is_what_matters(self):
        # With fusion on, contraction appears outside the div_rn lines; this proves
        # the previous test is sensitive to a dropped enable_fp_fusion=False.
        with open(ta.__file__) as handle:
            lines = handle.read().split("\n")
        for arch in AMD_ARCHS:
            with self.subTest(arch=arch):
                kernel = _compile(ta._ema_update_kernel, EMA_SIG, EMA_CASES["decay, device scalars"],
                                  arch, fusion=True)
                outside = [line for line in _fused_lines(kernel.asm["amdgcn"])
                           if not _on_div_rn(lines, line)]
                self.assertTrue(outside)

    def test_split_kernels_have_no_fma(self):
        for arch in AMD_ARCHS:
            for label, constexprs in SPLIT_CASES.items():
                with self.subTest(arch=arch, case=label):
                    kernel = _compile(ta._write_float_splits_kernel, SPLIT_SIG, constexprs, arch)
                    self.assertEqual(dict(_fused_lines(kernel.asm["amdgcn"])), {})

    def test_fp8_cast_kernel_compiles(self):
        for arch in AMD_ARCHS:
            for label, constexprs in CAST_CASES.items():
                with self.subTest(arch=arch, case=label):
                    kernel = _compile(ta._cast_fp8_centroids_kernel, CAST_SIG, constexprs, arch)
                    self.assertGreater(len(kernel.asm.get("hsaco", b"")), 0)

    def test_nvidia_sm90_emits_rn_instructions(self):
        constexprs = dict(DECAY=True, BLOCK_K=2, BLOCK_D=1024, SCALARS=False)
        sig = dict(EMA_SIG_HOST, **{name: "constexpr" for name in constexprs})
        kernel = triton.compile(
            ASTSource(fn=ta._ema_update_kernel, signature=sig, constexprs=constexprs),
            target=GPUTarget("cuda", 90, 32), options=dict(num_warps=4, enable_fp_fusion=False))
        ptx = kernel.asm["ptx"]
        self.assertNotIn("fma.rn.f32", ptx)
        for op in ("mul.rn.f32", "add.rn.f32", "sub.rn.f32", "div.rn.f32"):
            self.assertIn(op, ptx)


if __name__ == "__main__":
    unittest.main()
