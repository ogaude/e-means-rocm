# Automatic backend selection and default-cast rules on NVIDIA and AMD, with mocked
# devices. No GPU needed.

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from faiss.contrib.e_means import device_profile as dp
from faiss.contrib.e_means import e_means_torch
from faiss.contrib.e_means.torch_backend import (
    CuteBackend, TorchBackend, TritonBackend, select_backend)

X = SimpleNamespace(device=torch.device("cuda", 0))
K, D = 1024, 64


@contextmanager
def amd(arch, triton=True, out_dtype=True):
    dp._amd_arch.cache_clear()
    capability = mock.Mock(side_effect=AssertionError("capability queried on AMD"))
    with mock.patch.object(torch.version, "hip", "7.16"), \
         mock.patch.object(torch.cuda, "get_device_properties",
                           return_value=SimpleNamespace(gcnArchName=arch)), \
         mock.patch.object(torch.cuda, "get_device_capability", capability), \
         mock.patch.object(dp, "triton_available", return_value=triton), \
         mock.patch.object(dp, "addmm_out_dtype_available", return_value=out_dtype):
        yield
    dp._amd_arch.cache_clear()


@contextmanager
def nvidia(capability):
    with mock.patch.object(torch.version, "hip", None), \
         mock.patch.object(torch.cuda, "get_device_capability", return_value=capability):
        yield


class AmdSelectionTest(unittest.TestCase):

    def test_target_families_select_triton(self):
        for arch in ("gfx942", "gfx950", "gfx1100", "gfx1200", "gfx1201"):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(arch=arch, dtype=dtype), amd(arch):
                    self.assertIs(select_backend(X, dtype, K, D), TritonBackend)

    def test_cute_never_selected_on_amd(self):
        with amd("gfx942"):
            self.assertIs(select_backend(X, torch.bfloat16, K, D), TritonBackend)

    def test_unknown_family_fp32_falls_back_to_native(self):
        with amd("gfx90a"):
            self.assertIs(select_backend(X, torch.float32, K, D), TorchBackend)

    def test_unknown_family_reduced_precision_names_both_vendors(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), amd("gfx90a"):
                with self.assertRaisesRegex(RuntimeError, "SM80.*AMD gfx942"):
                    select_backend(X, dtype, K, D)

    def test_missing_triton_falls_back_to_native(self):
        with amd("gfx1200", triton=False):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.assertIs(select_backend(X, dtype, K, D), TorchBackend)

    def test_missing_triton_and_out_dtype_rejects_reduced_precision(self):
        with amd("gfx1200", triton=False, out_dtype=False):
            self.assertIs(select_backend(X, torch.float32, K, D), TorchBackend)
            with self.assertRaisesRegex(RuntimeError, "Triton installed"):
                select_backend(X, torch.bfloat16, K, D)

    def test_unknown_family_never_gets_native_reduced_precision(self):
        with amd("gfx90a", triton=False):
            with self.assertRaises(RuntimeError):
                select_backend(X, torch.bfloat16, K, D)

    def test_int8_selects_triton_on_target_families(self):
        for arch in ("gfx942", "gfx950", "gfx1100", "gfx1200", "gfx1201"):
            for dtype in (torch.int8, torch.uint8):
                with self.subTest(arch=arch, dtype=dtype), amd(arch):
                    self.assertIs(select_backend(X, dtype, K, D), TritonBackend)

    def test_int8_rejected_on_unknown_family_or_without_triton(self):
        for arch, triton in (("gfx90a", True), ("gfx1200", False)):
            with self.subTest(arch=arch, triton=triton), amd(arch, triton=triton):
                with self.assertRaisesRegex(RuntimeError, "INT8 assignment requires.*AMD gfx942"):
                    select_backend(X, torch.int8, K, D)

    def test_int8_dimension_bound_carries_over(self):
        with amd("gfx942"):
            with self.assertRaisesRegex(ValueError, "d <= 43919"):
                select_backend(X, torch.int8, K, 43920)

    def test_fp8_says_planned(self):
        with amd("gfx1200"):
            with self.assertRaisesRegex(RuntimeError,
                                        "float8_e4m3fn assignment requires.*planned"):
                select_backend(X, torch.float8_e4m3fn, K, D)

    def test_explicit_requests_pass_through(self):
        with amd("gfx90a"):
            self.assertIs(select_backend(X, torch.bfloat16, K, D, "triton"), TritonBackend)
            self.assertIs(select_backend(X, torch.bfloat16, K, D, "native"), TorchBackend)
            self.assertIs(select_backend(X, torch.bfloat16, K, D, "cute"), CuteBackend)


class NvidiaSelectionUnchangedTest(unittest.TestCase):

    def test_sm90_bf16_selects_cute(self):
        with nvidia((9, 0)):
            self.assertIs(select_backend(X, torch.bfloat16, K, D), CuteBackend)

    def test_int8_unchanged(self):
        for capability in ((8, 0), (9, 0), (10, 0)):
            with self.subTest(capability=capability), nvidia(capability):
                self.assertIs(select_backend(X, torch.int8, K, D), TritonBackend)
        with nvidia((8, 6)):
            with self.assertRaises(RuntimeError) as caught:
                select_backend(X, torch.int8, K, D)
        self.assertEqual(str(caught.exception),
                         "INT8 assignment requires an NVIDIA SM80, SM90, or SM100 GPU")

    def test_sm80_selects_triton(self):
        with nvidia((8, 0)):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                self.assertIs(select_backend(X, dtype, K, D), TritonBackend)

    def test_pre_sm80(self):
        with nvidia((7, 5)):
            self.assertIs(select_backend(X, torch.float32, K, D), TorchBackend)
            with self.assertRaises(RuntimeError) as caught:
                select_backend(X, torch.bfloat16, K, D)
        self.assertTrue(str(caught.exception).endswith("requires an NVIDIA SM80 or later GPU"))
        self.assertNotIn("AMD", str(caught.exception))

    def test_cpu(self):
        self.assertIs(select_backend(SimpleNamespace(device=torch.device("cpu")),
                                     torch.float32, K, D), TorchBackend)


class DefaultCastTest(unittest.TestCase):

    def test_nvidia_always_casts(self):
        with nvidia((7, 5)):
            for requested in ("auto", "native", "triton", "cute"):
                self.assertTrue(e_means_torch._default_cast_supported(X.device, requested))

    def test_amd_casts_when_a_backend_runs_bf16(self):
        cast = e_means_torch._default_cast_supported
        with amd("gfx1200"):
            for requested in ("auto", "triton", "native"):
                self.assertTrue(cast(X.device, requested), requested)
            self.assertFalse(cast(X.device, "cute"))
        with amd("gfx1200", triton=False):
            self.assertTrue(cast(X.device, "auto"))       # native BF16 fallback
            self.assertTrue(cast(X.device, "native"))
            self.assertFalse(cast(X.device, "triton"))
        with amd("gfx1200", out_dtype=False):
            self.assertTrue(cast(X.device, "auto"))       # Triton
            self.assertFalse(cast(X.device, "native"))
        with amd("gfx1200", triton=False, out_dtype=False):
            self.assertFalse(cast(X.device, "auto"))
        with amd("gfx90a"):
            for requested in ("auto", "triton", "native"):
                self.assertFalse(cast(X.device, requested), requested)


class TestSkipConditionsTest(unittest.TestCase):
    """tests/e_means_gpu_conditions.py: what each GPU test skip means per vendor."""

    def _conditions(self):
        import e_means_gpu_conditions
        return e_means_gpu_conditions

    def test_rocm_gfx_capability_is_not_nvidia_sm80(self):
        c = self._conditions()
        with amd("gfx1200"), mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "current_device", return_value=0), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)):
            self.assertFalse(c.nvidia_sm80())
            self.assertTrue(c.amd_triton_gpu())
            self.assertTrue(c.triton_gpu())

    def test_rocm_non_target_family(self):
        c = self._conditions()
        with amd("gfx90a"), mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "current_device", return_value=0):
            self.assertFalse(c.amd_triton_gpu())
            self.assertFalse(c.triton_gpu())

    def test_nvidia(self):
        c = self._conditions()
        with nvidia((8, 0)), mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertTrue(c.nvidia_sm80())
            self.assertFalse(c.amd_triton_gpu())
            self.assertTrue(c.graph_replay_gpu())
        with nvidia((7, 5)), mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertFalse(c.nvidia_sm80())

    def test_graph_replay_condition_on_rocm(self):
        c = self._conditions()
        with amd("gfx942"), mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.dict("os.environ", {dp.GRAPH_REPLAY_ENV: "0"}):
                self.assertFalse(c.graph_replay_gpu())
            with mock.patch.dict("os.environ", {dp.GRAPH_REPLAY_ENV: "1"}):
                self.assertTrue(c.graph_replay_gpu())

    def test_no_gpu(self):
        c = self._conditions()
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertFalse(c.nvidia_sm80() or c.amd_triton_gpu() or c.triton_gpu()
                             or c.graph_replay_gpu())


class GraphReplayPolicyTest(unittest.TestCase):

    def test_nvidia_always_allows_replay(self):
        with nvidia((9, 0)), mock.patch.dict("os.environ", {dp.GRAPH_REPLAY_ENV: "0"}):
            self.assertTrue(dp.graph_replay_supported(X.device))

    def test_rocm_default_is_eager_with_opt_in(self):
        with amd("gfx1200"):
            with mock.patch.dict("os.environ", {}, clear=False) as env:
                env.pop(dp.GRAPH_REPLAY_ENV, None)
                self.assertFalse(dp.graph_replay_supported(X.device))
            with mock.patch.dict("os.environ", {dp.GRAPH_REPLAY_ENV: "1"}):
                self.assertTrue(dp.graph_replay_supported(X.device))
            with mock.patch.dict("os.environ", {dp.GRAPH_REPLAY_ENV: "0"}):
                self.assertFalse(dp.graph_replay_supported(X.device))

    def test_training_takes_eager_path_on_rocm(self):
        model = e_means_torch.Emeans(D, K)
        x = SimpleNamespace(is_cuda=True, device=X.device)
        with amd("gfx1200"), mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop(dp.GRAPH_REPLAY_ENV, None)
            # Returns before reading the recipe, which is not set on a fresh model.
            self.assertFalse(model._replay_eligible(x))


class SplitKSelectionTest(unittest.TestCase):
    """Split-K choice: NVIDIA keeps _use_split_k; rdna4 caps it at 8 row blocks."""

    DEVICE = torch.device("cuda", 0)

    def setUp(self):
        try:
            from faiss.contrib.e_means import triton_assignment as ta
        except Exception as exc:  # pragma: no cover - environment without triton
            self.skipTest("triton not importable: %s" % exc)
        self.ta = ta

    def test_nvidia_matches_use_split_k(self):
        ta = self.ta
        with nvidia((9, 0)):
            for b in (256, 1024, 4096, 16384):
                for d in (64, 128, 768):
                    with self.subTest(b=b, d=d):
                        self.assertEqual(ta._select_split_k(self.DEVICE, b, 16384, d, True),
                                         ta._use_split_k(b, 16384, d, True))

    def test_rdna4_caps_row_blocks(self):
        ta = self.ta
        with amd("gfx1200"):
            self.assertTrue(ta._select_split_k(self.DEVICE, 256, 16384, 128, True))
            self.assertTrue(ta._select_split_k(self.DEVICE, 512, 16384, 128, True))
            for b in (1024, 4096):
                with self.subTest(b=b):
                    self.assertTrue(ta._use_split_k(b, 16384, 128, True))
                    self.assertFalse(ta._select_split_k(self.DEVICE, b, 16384, 128, True))

    def test_other_families_keep_use_split_k(self):
        ta = self.ta
        for arch in ("gfx942", "gfx950", "gfx1100"):
            with self.subTest(arch=arch), amd(arch):
                self.assertTrue(ta._select_split_k(self.DEVICE, 4096, 16384, 128, True))


if __name__ == "__main__":
    unittest.main()
