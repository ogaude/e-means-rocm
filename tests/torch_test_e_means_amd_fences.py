# W4 checks: NVIDIA-only Triton features stay fenced off on AMD. No GPU needed.
#
# * TMA tensor descriptors are never built on ROCm, even with an SM90-looking capability.
# * The device-property fields the Triton backend reads exist on this PyTorch build.
# * max_num_imprecise_acc (a Hopper FP8 hint, used only in FP8 branches) changes no AMD
#   instruction, so it needs no vendor guard.

import ast
import importlib.util
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

HAVE_TRITON = importlib.util.find_spec("triton") is not None


@unittest.skipUnless(HAVE_TRITON, "triton is not available")
class TmaFenceTest(unittest.TestCase):

    def _launch(self, hip):
        from faiss.contrib.e_means import triton_assignment as backend

        rows = torch.empty((8192, 256), dtype=torch.bfloat16)
        centroids = torch.empty((128, 256), dtype=torch.bfloat16)
        made = []

        def descriptor(tensor, block):
            made.append(block)
            return object()

        kernel = mock.MagicMock()
        with mock.patch.object(torch.version, "hip", hip), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)), \
             mock.patch.object(backend, "_assign_kernel", kernel), \
             mock.patch.object(backend, "TensorDescriptor", SimpleNamespace(from_tensor=descriptor)):
            backend._launch_assignment(rows, centroids, None, None, None, False)
        return made, kernel.__getitem__.return_value.call_args.kwargs

    def test_descriptors_on_nvidia_sm90(self):
        made, kwargs = self._launch(hip=None)
        self.assertEqual(len(made), 2)
        self.assertIsNotNone(kwargs.get("x_desc"))

    def test_no_descriptors_on_rocm(self):
        made, kwargs = self._launch(hip="7.16")
        self.assertEqual(made, [])
        self.assertIsNone(kwargs.get("x_desc"))


class DevicePropertyFieldsTest(unittest.TestCase):

    def test_fields_used_by_the_triton_backend_exist(self):
        fields = set(dir(torch._C._CudaDeviceProperties))
        self.assertIn("multi_processor_count", fields)   # EMA grid sizing
        if torch.version.hip is not None:
            self.assertIn("gcnArchName", fields)          # device_profile arch source


@unittest.skipUnless(HAVE_TRITON, "triton is not available")
class ImpreciseAccHintTest(unittest.TestCase):
    """Compile the FP8 assignment kernel with and without the hint; compare instructions."""

    def _kernels(self):
        from faiss.contrib.e_means import triton_assignment as ta

        source = open(ta.__file__).read()
        tree = ast.parse(source)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        directory = scratch.name
        modules = {}
        for label, strip in (("hint", False), ("stripped", True)):
            body = "import triton\nimport triton.language as tl\n"
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in (
                        "_selected_fp8_distance", "_assign_kernel"):
                    text = "".join("@" + ast.get_source_segment(source, d) + "\n"
                                   for d in node.decorator_list)
                    text += ast.get_source_segment(source, node)
                    if strip:
                        text = re.sub(r",?\s*max_num_imprecise_acc=max\(32, BLOCK_D\),?", ",", text)
                    body += "\n\n" + text + "\n"
            name = "emeans_fence_%s" % label
            with open("%s/%s.py" % (directory, name), "w") as handle:
                handle.write(body)
            modules[label] = name
            self.addCleanup(sys.modules.pop, name, None)
        sys.path.insert(0, directory)
        try:
            return {label: importlib.import_module(name) for label, name in modules.items()}
        finally:
            sys.path.remove(directory)

    SIG = dict(x_ptr="*fp8e4nv", centroid_ptr="*fp8e4nv", centroid_norm_ptr="*fp32",
               distance_ptr="*fp32", label_ptr="*i64", B="i32", K="i32", D="i32",
               stride_xb="i64", stride_xd="i64", stride_ck="i64", stride_cd="i64")
    CX = dict(BLOCK_B=64, BLOCK_K=256, BLOCK_D=64, FULL_D=False, REDUCED=True, FP16=False,
              FP8=True, WIDE_CENTROID=False, x_desc=None, centroid_desc=None, SPHERICAL=False)

    def _streams(self, modules, target, asm_key, pattern):
        import triton
        from triton.compiler import ASTSource

        sig = dict(self.SIG, **{name: "constexpr" for name in self.CX})
        streams = {}
        for label, module in modules.items():
            kernel = triton.compile(
                ASTSource(fn=module._assign_kernel, signature=sig, constexprs=self.CX),
                target=target, options=dict(num_warps=8, num_stages=2, enable_fp_fusion=False))
            streams[label] = [line.strip() for line in kernel.asm[asm_key].split("\n")
                              if re.match(pattern, line) and not line.strip().startswith(".")]
        return streams

    def test_hint_changes_no_amd_instruction(self):
        from triton.backends.compiler import GPUTarget

        modules = self._kernels()
        self.assertIn("max_num_imprecise_acc", open(modules["hint"].__file__).read())
        self.assertNotIn("max_num_imprecise_acc", open(modules["stripped"].__file__).read())
        for arch in ("gfx942", "gfx1200"):
            with self.subTest(arch=arch):
                target = GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64)
                streams = self._streams(modules, target, "amdgcn", r"\s+[vsgd][a-z_0-9]+\s")
                self.assertGreater(len(streams["hint"]), 1000)
                self.assertEqual(streams["hint"], streams["stripped"])
        # Positive control: on sm90 the hint is real (FP8 WGMMA accumulation), so the
        # same comparison must see a difference there.
        streams = self._streams(modules, GPUTarget("cuda", 90, 32), "ptx", r"\s+[a-z]")
        self.assertNotEqual(streams["hint"], streams["stripped"])


@unittest.skipUnless(HAVE_TRITON, "triton is not available")
class AmdTuningTableTest(unittest.TestCase):
    """_AMD_DIRECT_CONFIGS (W6): shape of the rules, lookup, and LDS budget."""

    def setUp(self):
        from faiss.contrib.e_means import device_profile as dp
        from faiss.contrib.e_means import triton_assignment as ta
        self.dp, self.ta = dp, ta
        dp._amd_arch.cache_clear()
        self.addCleanup(dp._amd_arch.cache_clear)

    def _on(self, arch):
        return mock.patch.multiple(
            torch.cuda, get_device_properties=mock.Mock(
                return_value=SimpleNamespace(gcnArchName=arch)))

    def test_rule_shapes_match_the_full_d_path(self):
        ta = self.ta
        for family, rules in ta._AMD_DIRECT_CONFIGS.items():
            self.assertIn(family, ("cdna3", "cdna4", "rdna3", "rdna4"))
            for kind, entries in rules.items():
                self.assertIsNone(entries[-1][0], "%s/%s needs a catch-all rule" % (family, kind))
                for d in range(1, 2049):
                    config = next(c for max_d, c in entries if max_d is None or d <= max_d)
                    full_d = (kind == "reduced"
                              and ta._DIRECT_REDUCED_BLOCK_D < d <= ta._FULL_D_MAX_D)
                    with self.subTest(family=family, kind=kind, d=d):
                        self.assertEqual(config[2] is None, full_d)

    def test_nvidia_and_untuned_families_use_the_constants(self):
        ta = self.ta
        with mock.patch.object(torch.version, "hip", None):
            self.assertIsNone(ta._amd_direct_config(torch.device("cuda", 0), True, 64))
        for arch in ("gfx942", "gfx950", "gfx1100", "gfx90a"):
            self.dp._amd_arch.cache_clear()
            with self.subTest(arch=arch), mock.patch.object(torch.version, "hip", "7.16"), \
                 self._on(arch):
                expected = self.dp.family_of(arch) in ta._AMD_DIRECT_CONFIGS
                got = ta._amd_direct_config(torch.device("cuda", 0), True, 64)
                self.assertEqual(got is not None, expected)

    def test_rdna4_lookup(self):
        ta = self.ta
        with mock.patch.object(torch.version, "hip", "7.16"), self._on("gfx1201"):
            look = lambda reduced, d: ta._amd_direct_config(torch.device("cuda", 0), reduced, d)
            self.assertEqual(look(True, 32), (128, 64, 32, 4, 2))
            self.assertEqual(look(True, 64), (64, 128, 32, 4, 2))
            self.assertEqual(look(True, 96)[2], None)
            self.assertEqual(look(True, 768), (64, 256, 64, 8, 2))
            self.assertEqual(look(False, 768), (64, 64, 16, 4, 3))

    def test_rdna4_rules_fit_lds(self):
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        ta = self.ta
        budget = self.dp.profile_for_arch("gfx1200").lds_bytes
        sig = dict(x_ptr="*bf16", centroid_ptr="*bf16", centroid_norm_ptr="*fp32",
                   distance_ptr="*fp32", label_ptr="*i64", B="i32", K="i32", D="i32",
                   stride_xb="i64", stride_xd="i64", stride_ck="i64", stride_cd="i64")
        for kind, d in (("reduced", 32), ("reduced", 64), ("reduced", 128), ("reduced", 768),
                        ("fp32", 32), ("fp32", 768)):
            config = next(c for max_d, c in ta._AMD_DIRECT_CONFIGS["rdna4"][kind]
                          if max_d is None or d <= max_d)
            block_b, block_k, block_d, warps, stages = config
            full = block_d is None
            dtype = "bf16" if kind == "reduced" else "fp32"
            s = dict(sig, x_ptr="*" + dtype, centroid_ptr="*" + dtype)
            cx = dict(BLOCK_B=block_b, BLOCK_K=block_k,
                      BLOCK_D=triton.next_power_of_2(d) if full else block_d, FULL_D=full,
                      REDUCED=kind == "reduced", FP16=False, FP8=False, WIDE_CENTROID=False,
                      x_desc=None, centroid_desc=None, SPHERICAL=False)
            s.update({name: "constexpr" for name in cx})
            with self.subTest(kind=kind, d=d):
                kernel = triton.compile(
                    ASTSource(fn=ta._assign_kernel, signature=s, constexprs=cx),
                    target=GPUTarget("hip", "gfx1200", 32),
                    options=dict(num_warps=warps, num_stages=stages, enable_fp_fusion=False))
                self.assertLessEqual(kernel.metadata.shared, budget)


if __name__ == "__main__":
    unittest.main()
