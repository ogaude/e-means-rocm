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
def amd(arch, triton=True):
    dp._amd_arch.cache_clear()
    capability = mock.Mock(side_effect=AssertionError("capability queried on AMD"))
    with mock.patch.object(torch.version, "hip", "7.16"), \
         mock.patch.object(torch.cuda, "get_device_properties",
                           return_value=SimpleNamespace(gcnArchName=arch)), \
         mock.patch.object(torch.cuda, "get_device_capability", capability), \
         mock.patch.object(dp, "triton_available", return_value=triton):
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

    def test_missing_triton_falls_back_to_native_fp32(self):
        with amd("gfx1200", triton=False):
            self.assertIs(select_backend(X, torch.float32, K, D), TorchBackend)
            with self.assertRaisesRegex(RuntimeError, "Triton installed"):
                select_backend(X, torch.bfloat16, K, D)

    def test_int8_and_fp8_say_planned(self):
        for dtype, message in ((torch.int8, "INT8 assignment requires.*planned"),
                               (torch.uint8, "INT8 assignment requires.*planned"),
                               (torch.float8_e4m3fn, "float8_e4m3fn assignment requires.*planned")):
            with self.subTest(dtype=dtype), amd("gfx1200"):
                with self.assertRaisesRegex(RuntimeError, message):
                    select_backend(X, dtype, K, D)

    def test_explicit_requests_pass_through(self):
        with amd("gfx90a"):
            self.assertIs(select_backend(X, torch.bfloat16, K, D, "triton"), TritonBackend)
            self.assertIs(select_backend(X, torch.bfloat16, K, D, "native"), TorchBackend)
            self.assertIs(select_backend(X, torch.bfloat16, K, D, "cute"), CuteBackend)


class NvidiaSelectionUnchangedTest(unittest.TestCase):

    def test_sm90_bf16_selects_cute(self):
        with nvidia((9, 0)):
            self.assertIs(select_backend(X, torch.bfloat16, K, D), CuteBackend)

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

    def test_amd_casts_only_for_triton(self):
        with amd("gfx1200"):
            self.assertTrue(e_means_torch._default_cast_supported(X.device, "auto"))
            self.assertTrue(e_means_torch._default_cast_supported(X.device, "triton"))
            self.assertFalse(e_means_torch._default_cast_supported(X.device, "native"))
        with amd("gfx90a"):
            self.assertFalse(e_means_torch._default_cast_supported(X.device, "auto"))
        with amd("gfx1200", triton=False):
            self.assertFalse(e_means_torch._default_cast_supported(X.device, "auto"))


if __name__ == "__main__":
    unittest.main()
