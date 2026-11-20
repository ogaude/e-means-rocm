# Launch configurations of the Triton assignment kernels, GPU-free.
#
# What this guards:
#   * every configuration the launcher can pick for each AMD target family (direct,
#     split-K and its reduction; FP32, BF16, FP8 where native) compiles within the
#     family's LDS budget (device_profile), as the README's "compile-checked" promises;
#   * the configurations themselves: NVIDIA keeps the upstream constants, rdna4 its
#     tuned tables, cdna4 its 64 KiB cap, other families the constants;
#   * _launch_assignment passes exactly _direct_config's choice to the kernel, and the
#     per-family split-K stage count;
#   * rule boundaries: INT8 small-batch rows and widths, the split-K row-block cap;
#   * the Triton fallback of the AMD arch lookup.

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

try:
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from faiss.contrib.e_means import device_profile as dp
    from faiss.contrib.e_means import triton_assignment as ta
    HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    HAVE_TRITON = False

ARCHS = {"cdna3": "gfx942", "cdna4": "gfx950", "rdna3": "gfx1100", "rdna4": "gfx1200"}
WIDTHS = (32, 64, 96, 128, 192, 256, 768)
DTYPE = {"fp32": "fp32", "bf16": "bf16", "fp8": "fp8e4nv"}
ASSIGN_SIG = dict(x_ptr="*bf16", centroid_ptr="*bf16", centroid_norm_ptr="*fp32",
                  distance_ptr="*fp32", label_ptr="*i64", B="i32", K="i32", D="i32",
                  stride_xb="i64", stride_xd="i64", stride_ck="i64", stride_cd="i64")
SPLIT_SIG = dict(x_ptr="*bf16", centroid_ptr="*bf16", centroid_norm_ptr="*fp32",
                 partial_score_ptr="*fp32", partial_label_ptr="*i32", B="i32", K="i32",
                 D="i32", stride_xb="i64", stride_xd="i64", stride_ck="i64",
                 stride_cd="i64")
REDUCE_SIG = dict(x_ptr="*bf16", centroid_ptr="*bf16", partial_score_ptr="*fp32",
                  partial_label_ptr="*i32", distance_ptr="*fp32", label_ptr="*i64",
                  B="i32", D="i32", stride_xb="i64", stride_xd="i64", stride_ck="i64",
                  stride_cd="i64")


def _target(arch):
    return GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64)


def _compile(fn, signature, constexprs, arch, warps, stages):
    signature = dict(signature, **{name: "constexpr" for name in constexprs})
    return triton.compile(ASTSource(fn=fn, signature=signature, constexprs=constexprs),
                          target=_target(arch),
                          options=dict(num_warps=warps, num_stages=stages,
                                       enable_fp_fusion=False))


def _kinds(family):
    kinds = ["fp32", "bf16"]
    if dp._FAMILIES[family]["native_fp8"]:
        kinds.append("fp8")
    return kinds


@unittest.skipUnless(HAVE_TRITON, "triton not installed")
class LdsBudgetTest(unittest.TestCase):
    """What the launcher picks must fit the family's LDS budget, kernel by kernel."""

    def test_direct_kernel_fits(self):
        for family, arch in ARCHS.items():
            budget = dp.profile_for_arch(arch).lds_bytes
            for kind in _kinds(family):
                for d in WIDTHS:
                    full_d, bb, bk, bd, warps, stages = ta._direct_config(
                        family, kind != "fp32", kind == "fp8", d)
                    cx = dict(BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=bd, FULL_D=full_d,
                              REDUCED=kind != "fp32", FP16=False, FP8=kind == "fp8",
                              WIDE_CENTROID=False, x_desc=None, centroid_desc=None,
                              SPHERICAL=False)
                    sig = dict(ASSIGN_SIG, x_ptr="*" + DTYPE[kind],
                               centroid_ptr="*" + DTYPE[kind])
                    with self.subTest(arch=arch, kind=kind, d=d):
                        kernel = _compile(ta._assign_kernel, sig, cx, arch, warps, stages)
                        self.assertLessEqual(kernel.metadata.shared, budget)

    def test_split_k_kernels_fit(self):
        for family, arch in ARCHS.items():
            budget = dp.profile_for_arch(arch).lds_bytes
            stages = ta._AMD_SPLIT_K_STAGES.get(family, ta._SPLIT_K_NUM_STAGES)
            for kind in _kinds(family):
                if kind == "fp32":
                    continue  # split-K is reduced precision only
                dtype = DTYPE[kind]
                with self.subTest(arch=arch, kind=kind, kernel="split"):
                    cx = dict(PARTITIONS=ta._SPLIT_K_MAX_PARTITIONS,
                              BLOCK_B=ta._SPLIT_K_BLOCK_B, BLOCK_K=ta._SPLIT_K_BLOCK_K,
                              BLOCK_D=ta._SPLIT_K_BLOCK_D, FP16=False, FP8=kind == "fp8",
                              WIDE_CENTROID=False)
                    sig = dict(SPLIT_SIG, x_ptr="*" + dtype, centroid_ptr="*" + dtype)
                    kernel = _compile(ta._assign_split_k_kernel, sig, cx, arch,
                                      ta._SPLIT_K_NUM_WARPS, stages)
                    self.assertLessEqual(kernel.metadata.shared, budget)
                with self.subTest(arch=arch, kind=kind, kernel="reduce"):
                    p = ta._SPLIT_K_MAX_PARTITIONS
                    cx = dict(PARTITIONS=p, BLOCK_PARTITIONS=triton.next_power_of_2(p),
                              BLOCK_D=128, FP16=False, FP8=kind == "fp8", SPHERICAL=False)
                    sig = dict(REDUCE_SIG, x_ptr="*" + dtype, centroid_ptr="*" + dtype)
                    kernel = _compile(ta._reduce_split_k_kernel, sig, cx, arch, 4, 1)
                    self.assertLessEqual(kernel.metadata.shared, budget)


@unittest.skipUnless(HAVE_TRITON, "triton not installed")
class DirectConfigTest(unittest.TestCase):
    """Pinned launch configurations: (FULL_D, BLOCK_B, BLOCK_K, BLOCK_D, warps, stages)."""

    def test_nvidia_and_untuned_families_keep_the_constants(self):
        s = ta._NUM_STAGES
        expected = {
            (True, False, 64): (False, 64, 256, 64, 8, s),
            (True, False, 96): (True, 64, 128, 128, 4, s),
            (True, False, 768): (False, 128, 256, 64, 8, s),    # aligned: 128-row tile
            (False, False, 768): (False, 64, 128, 32, 8, s),
            (True, True, 64): (False, 64, 256, 64, 8, s),       # FP8: 128 would pad
            (True, True, 128): (True, 64, 128, 128, 4, s),
            (True, True, 768): (False, 64, 256, 128, 8, s),     # FP8: wider D, no pad
        }
        for family in (None, "cdna3", "rdna3"):
            for (reduced, fp8, d), config in expected.items():
                with self.subTest(family=family, reduced=reduced, fp8=fp8, d=d):
                    self.assertEqual(ta._direct_config(family, reduced, fp8, d), config)
        # Misaligned tensors keep the 64-row tile at d > 128.
        self.assertEqual(ta._direct_config(None, True, False, 768, wide_rows_ok=False),
                         (False, 64, 256, 64, 8, s))

    def test_rdna4_tables(self):
        cases = {
            (True, False, 64): (False, 32, 256, 32, 4, 2),
            (True, False, 96): (True, 128, 64, 128, 8, 1),
            (True, False, 768): (False, 64, 256, 64, 8, 2),
            (False, False, 768): (False, 64, 64, 16, 4, 3),
            (True, True, 64): (False, 32, 256, 64, 4, 1),
            (True, True, 65): (True, 32, 256, 128, 4, 1),
            (True, True, 128): (True, 32, 256, 128, 4, 1),
            (True, True, 129): (False, 64, 256, 64, 8, 1),
        }
        for (reduced, fp8, d), config in cases.items():
            with self.subTest(reduced=reduced, fp8=fp8, d=d):
                self.assertEqual(ta._direct_config("rdna4", reduced, fp8, d), config)

    def test_cdna4_caps_the_128_row_tile(self):
        s = ta._NUM_STAGES
        self.assertEqual(ta._direct_config("cdna4", True, False, 768),
                         (False, 64, 256, 64, 8, s))
        self.assertEqual(ta._direct_config("cdna4", True, False, 96),
                         (True, 64, 128, 128, 4, s))
        self.assertEqual(ta._AMD_SPLIT_K_STAGES.get("cdna4"), 1)

    def test_family_without_fp8_rules_uses_the_constants(self):
        rules = {"reduced": ta._AMD_DIRECT_CONFIGS["rdna4"]["reduced"]}
        with mock.patch.dict(ta._AMD_DIRECT_CONFIGS, {"rdna4": rules}):
            self.assertEqual(ta._direct_config("rdna4", True, True, 768),
                             ta._direct_config(None, True, True, 768))


@unittest.skipUnless(HAVE_TRITON, "triton not installed")
class LaunchSiteTest(unittest.TestCase):
    """_launch_assignment passes _direct_config's choice to the kernel unchanged."""

    def _launch(self, family, dtype, d, split_k=False, b=8192, k=4096):
        x = torch.zeros((b, d), dtype=dtype)
        c = torch.zeros((k, d), dtype=dtype)
        norm = torch.zeros(k)
        dist, labels = torch.zeros(b), torch.zeros(b, dtype=torch.int64)
        kernels = {name: mock.MagicMock() for name in
                   ("_assign_kernel", "_assign_split_k_kernel", "_reduce_split_k_kernel")}
        with mock.patch.object(ta, "_amd_family", return_value=family), \
             mock.patch.object(ta, "_select_split_k", return_value=split_k), \
             mock.patch.multiple(ta, **kernels):
            ta._launch_assignment(x, c, norm, dist, labels, False)
        return kernels

    def test_direct_launch_uses_direct_config(self):
        for family in (None, "rdna4", "cdna4", "cdna3"):
            for dtype, d in ((torch.bfloat16, 64), (torch.bfloat16, 96),
                             (torch.bfloat16, 768), (torch.float32, 768),
                             (torch.float8_e4m3fn, 64), (torch.float8_e4m3fn, 128),
                             (torch.float8_e4m3fn, 768)):
                with self.subTest(family=family, dtype=dtype, d=d):
                    kernels = self._launch(family, dtype, d)
                    launch = kernels["_assign_kernel"].__getitem__.return_value
                    kw = launch.call_args.kwargs
                    got = (kw["FULL_D"], kw["BLOCK_B"], kw["BLOCK_K"], kw["BLOCK_D"],
                           kw["num_warps"], kw["num_stages"])
                    reduced, fp8 = dtype != torch.float32, dtype == torch.float8_e4m3fn
                    # CPU tensors from torch.zeros are aligned, so the 128-row tile is allowed.
                    self.assertEqual(got, ta._direct_config(family, reduced, fp8, d))
                    self.assertEqual(kw["FP8"], fp8)

    def test_split_k_uses_the_family_stage_count(self):
        for family, stages in (("cdna4", 1), ("rdna4", ta._SPLIT_K_NUM_STAGES),
                               (None, ta._SPLIT_K_NUM_STAGES)):
            with self.subTest(family=family):
                kernels = self._launch(family, torch.bfloat16, 128, split_k=True, b=256)
                launch = kernels["_assign_split_k_kernel"].__getitem__.return_value
                self.assertEqual(launch.call_args.kwargs["num_stages"], stages)
                kernels["_assign_kernel"].__getitem__.assert_not_called()


@unittest.skipUnless(HAVE_TRITON, "triton not installed")
class RuleBoundaryTest(unittest.TestCase):

    def test_int8_small_batch_boundaries(self):
        cfg = ta._int8_launch_config
        cases = {
            (128, 4096): (True, 64, 256, 128, 8, 1),     # small-batch rule, last row
            (128, 4097): (True, 128, 64, 128, 4, 1),     # main table
            (144, 2048): (True, 64, 128, 256, 8, 1),
            (144, 2049): (True, 128, 64, 256, 8, 1),
            (256, 2048): (True, 64, 128, 256, 8, 1),
            (256, 2049): (True, 128, 64, 256, 8, 1),
            (272, 1024): (False, 64, 256, 128, 8, 1),    # tiled path, no small rule
            (16, 4097): (True, 128, 128, 16, 4, 1),
        }
        for (physical_d, rows), expected in cases.items():
            with self.subTest(physical_d=physical_d, rows=rows):
                self.assertEqual(cfg(physical_d, "rdna4", rows), expected)
        self.assertEqual(cfg(128, "rdna4", None), cfg(128, "rdna4", 4097))

    def test_int8_nvidia_ignores_rows(self):
        for physical_d in (16, 128, 256, 512):
            with self.subTest(physical_d=physical_d):
                self.assertEqual(ta._int8_launch_config(physical_d, None, 1024),
                                 ta._int8_launch_config(physical_d, None))
                self.assertEqual(ta._int8_launch_config(physical_d, None)[5], 3)

    def test_split_k_cap_boundary(self):
        device = torch.device("cuda", 0)
        with mock.patch.object(torch.version, "hip", "7.16"), \
             mock.patch.object(dp, "device_profile",
                               return_value=dp.profile_for_arch("gfx1200")):
            self.assertTrue(ta._use_split_k(513, 16384, 128, True))
            self.assertTrue(ta._select_split_k(device, 512, 16384, 128, True))
            self.assertFalse(ta._select_split_k(device, 513, 16384, 128, True))
            self.assertFalse(ta._select_split_k(device, 256, 16384, 128, False))  # FP32


class AmdArchFallbackTest(unittest.TestCase):
    """_amd_arch falls back to the Triton target when gcnArchName is empty."""

    def setUp(self):
        from faiss.contrib.e_means import device_profile
        self.dp = device_profile
        self.dp._amd_arch.cache_clear()
        self.addCleanup(self.dp._amd_arch.cache_clear)

    def _props(self):
        return mock.patch.object(torch.cuda, "get_device_properties",
                                 return_value=SimpleNamespace(gcnArchName=""))

    @unittest.skipUnless(HAVE_TRITON, "triton not installed")
    def test_triton_target_is_used(self):
        target = SimpleNamespace(arch="gfx1201")
        driver = SimpleNamespace(get_current_target=lambda: target)
        with self._props(), mock.patch.object(triton.runtime.driver, "_active", driver,
                                              create=True), \
             mock.patch.object(type(triton.runtime.driver), "active",
                               new_callable=mock.PropertyMock, return_value=driver):
            self.assertEqual(self.dp._amd_arch(0), "gfx1201")
            self.assertEqual(self.dp.family_of(self.dp._amd_arch(0)), "rdna4")

    @unittest.skipUnless(HAVE_TRITON, "triton not installed")
    def test_no_arch_means_no_family(self):
        def broken():
            raise RuntimeError("no driver")
        driver = SimpleNamespace(get_current_target=broken)
        with self._props(), \
             mock.patch.object(type(triton.runtime.driver), "active",
                               new_callable=mock.PropertyMock, return_value=driver):
            self.assertIsNone(self.dp._amd_arch(0))
            self.assertIsNone(self.dp.family_of(self.dp._amd_arch(0)))


if __name__ == "__main__":
    unittest.main()
