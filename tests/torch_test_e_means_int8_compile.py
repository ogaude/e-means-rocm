# Offline compile checks for INT8 assignment on the four AMD target families. No GPU
# is needed: the real _assign_int8_kernel is compiled with the configuration
# _int8_launch_config picks for each family, and the compiler's report is inspected.
#
# What this guards:
#   * every INT8 launch configuration stays inside the family's LDS budget
#     (upstream's num_stages=3 does not, outside gfx942);
#   * each stage count in _AMD_INT8_STAGES is the largest (up to 3) that fits, so the
#     table is neither stale nor needlessly conservative for this Triton;
#   * the kernel uses the family's integer matrix instruction (MFMA i8 / WMMA iu8);
#   * NVIDIA keeps the upstream launch configuration.

import re
import unittest

try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from faiss.contrib.e_means import triton_assignment as ta
    from faiss.contrib.e_means.device_profile import profile_for_arch
    HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    HAVE_TRITON = False

# One representative architecture per family.
ARCHS = {"cdna3": "gfx942", "cdna4": "gfx950", "rdna3": "gfx1100", "rdna4": "gfx1200"}
# physical_d values covering the three launch paths: FULL_D with BLOCK_D 128 and 256,
# and the tiled path.
PATHS = {"full128": 128, "full256": 256, "tiled": 512}
MATRIX = re.compile(r"^\s+(v_mfma_i32_\w+_i8|v_wmma_i32_\w+_iu8)", re.M)
SIGNATURE = dict(x_ptr="*i8", centroid_ptr="*i8", centroid_norm_ptr="*i32",
                 distance_ptr="*fp32", label_ptr="*i64", B="i32", K="i32", D="i32",
                 stride_xb="i64", stride_xd="i64", stride_ck="i64", stride_cd="i64")


def _compile(arch, physical_d, num_stages=None, unsigned=False, packed=True):
    family = profile_for_arch(arch).family
    full_d, block_b, block_k, block_d, num_warps, stages = ta._int8_launch_config(
        physical_d, family)
    constexprs = dict(UNSIGNED=unsigned, BLOCK_B=block_b, BLOCK_K=block_k, BLOCK_D=block_d,
                      WIDE_CENTROID=False, FULL_D=full_d, PACKED=packed,
                      x_desc=None, centroid_desc=None)
    signature = dict(SIGNATURE, **{name: "constexpr" for name in constexprs})
    return triton.compile(
        ASTSource(fn=ta._assign_int8_kernel, signature=signature, constexprs=constexprs),
        target=GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64),
        options=dict(num_warps=num_warps,
                     num_stages=stages if num_stages is None else num_stages))


@unittest.skipUnless(HAVE_TRITON, "triton not installed")
class Int8LaunchConfigTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ta._load_int8_libdevice()

    def test_configurations_fit_the_lds_budget(self):
        for family, arch in ARCHS.items():
            budget = profile_for_arch(arch).lds_bytes
            for path, physical_d in PATHS.items():
                for unsigned in (False, True):
                    for packed in (True, False):
                        with self.subTest(arch=arch, path=path, unsigned=unsigned,
                                          packed=packed):
                            kernel = _compile(arch, physical_d, unsigned=unsigned,
                                              packed=packed)
                            self.assertLessEqual(kernel.metadata.shared, budget)
                            self.assertTrue(MATRIX.search(kernel.asm["amdgcn"]),
                                            "no integer matrix instruction")

    def test_stage_counts_are_the_largest_that_fit(self):
        for family, arch in ARCHS.items():
            budget = profile_for_arch(arch).lds_bytes
            for path, physical_d in PATHS.items():
                stages = ta._int8_launch_config(physical_d, family)[5]
                if stages == 3:
                    continue
                with self.subTest(arch=arch, path=path, stages=stages):
                    more = _compile(arch, physical_d, num_stages=stages + 1)
                    self.assertGreater(more.metadata.shared, budget)

    def test_upstream_hard_coded_stages_overflow_outside_cdna3(self):
        # The reason for the table: three stages on the tiled path.
        for family, arch in ARCHS.items():
            with self.subTest(arch=arch):
                shared = _compile(arch, PATHS["tiled"], num_stages=3).metadata.shared
                budget = profile_for_arch(arch).lds_bytes
                if family == "cdna3":
                    self.assertLessEqual(shared, budget)
                else:
                    self.assertGreater(shared, budget)

    def test_nvidia_keeps_upstream_configuration(self):
        self.assertEqual(ta._int8_launch_config(128), (True, 128, 128, 128, 4, 3))
        self.assertEqual(ta._int8_launch_config(256), (True, 128, 128, 256, 4, 3))
        self.assertEqual(ta._int8_launch_config(512), (False, ta._INT8_BLOCK_B,
                                                       ta._INT8_BLOCK_K, ta._INT8_BLOCK_D,
                                                       8, 3))

    def test_every_family_has_an_entry(self):
        self.assertEqual(set(ta._AMD_INT8_STAGES), set(ARCHS))


if __name__ == "__main__":
    unittest.main()
