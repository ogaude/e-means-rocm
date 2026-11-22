# supported(), memory_estimate() and check_finite. The mocked-device tests need no
# GPU; the others compare the answers with real training on the local device.

import unittest
import warnings
from unittest import mock

import torch

from faiss.contrib.e_means import Emeans, QuantizedEmeans, memory_estimate, supported
from faiss.contrib.e_means.torch_backend import CuteBackend, TritonBackend
from faiss.contrib.e_means.quantizer import CastQuantizer
from torch_test_e_means_backend_selection import amd, nvidia
from e_means_gpu_conditions import triton_gpu

FP8 = torch.float8_e4m3fn
FLOATING = (torch.float32, torch.float16, torch.bfloat16)
BYTES = (torch.int8, torch.uint8)
ALL = FLOATING + BYTES + (FP8,)
BACKENDS = ("auto", "native", "triton", "cute")
GPU = torch.cuda.is_available()


def _software_present():
    """Treat Triton and the CuTe stack as installed, for mocked devices."""
    return mock.patch.multiple(
        "faiss.contrib.e_means.torch_backend",
        TritonBackend=type("T", (TritonBackend,), {"require_training": lambda self, d: None}),
        CuteBackend=type("C", (CuteBackend,), {"require_training": lambda self, d: None}),
    )


class SupportedMockedTest(unittest.TestCase):

    def test_cpu_runs_everything_on_native(self):
        for dtype in ALL:
            for spherical in (False, True):
                with self.subTest(dtype=dtype, spherical=spherical):
                    self.assertEqual(supported("cpu", dtype, spherical=spherical), "native")
                    self.assertEqual(supported("cpu", dtype, backend="native"), "native")
                    self.assertIsNone(supported("cpu", dtype, backend="triton"))
                    self.assertIsNone(supported("cpu", dtype, backend="cute"))

    def test_unknown_representation_and_backend(self):
        self.assertIsNone(supported("cpu", torch.float64))
        with self.assertRaises(ValueError):
            supported("cpu", torch.float32, backend="cuda")

    def test_amd_target_families(self):
        with _software_present():
            for arch, fp8 in (("gfx942", False), ("gfx950", True), ("gfx1100", False),
                              ("gfx1200", True)):
                with self.subTest(arch=arch), amd(arch):
                    for dtype in FLOATING + BYTES:
                        self.assertEqual(supported("cuda:0", dtype), "triton")
                    self.assertEqual(supported("cuda:0", FP8), "triton" if fp8 else None)
                    for dtype in BYTES + (FP8,):
                        self.assertIsNone(supported("cuda:0", dtype, spherical=True))
                        self.assertIsNone(supported("cuda:0", dtype, backend="native"))
                    for dtype in FLOATING:
                        self.assertEqual(supported("cuda:0", dtype, spherical=True), "triton")
                        self.assertEqual(supported("cuda:0", dtype, backend="native"), "native")
                    self.assertIsNone(supported("cuda:0", torch.bfloat16, backend="cute"))
                    self.assertIsNone(supported("cuda:0", torch.int8, d=43920))

    def test_amd_outside_targets_or_without_triton(self):
        with amd("gfx90a"):
            self.assertEqual(supported("cuda:0", torch.float32), "native")
            self.assertIsNone(supported("cuda:0", torch.bfloat16))
            self.assertIsNone(supported("cuda:0", torch.int8))
            self.assertIsNone(supported("cuda:0", torch.float32, backend="triton"))
        with amd("gfx1200", triton=False):
            self.assertEqual(supported("cuda:0", torch.bfloat16), "native")
            self.assertIsNone(supported("cuda:0", torch.int8))
            self.assertIsNone(supported("cuda:0", torch.bfloat16, backend="triton"))

    def test_nvidia(self):
        with _software_present():
            with nvidia((9, 0)):
                self.assertEqual(supported("cuda:0", torch.bfloat16), "cute")
                self.assertEqual(supported("cuda:0", torch.float16), "triton")
                self.assertEqual(supported("cuda:0", FP8), "triton")
                self.assertEqual(supported("cuda:0", FP8, backend="cute"), "cute")
                self.assertEqual(supported("cuda:0", torch.int8, backend="cute"), "cute")
                self.assertIsNone(supported("cuda:0", torch.float16, backend="cute"))
                self.assertIsNone(supported("cuda:0", torch.float32, backend="cute"))
                self.assertEqual(supported("cuda:0", torch.bfloat16, spherical=True,
                                           backend="cute"), "cute")
                self.assertIsNone(supported("cuda:0", FP8, spherical=True, backend="cute"))
            with nvidia((8, 0)):
                self.assertEqual(supported("cuda:0", torch.bfloat16), "triton")
                self.assertEqual(supported("cuda:0", torch.int8), "triton")
                self.assertIsNone(supported("cuda:0", FP8))
                self.assertIsNone(supported("cuda:0", torch.bfloat16, backend="cute"))
            with nvidia((8, 9)):
                self.assertIsNone(supported("cuda:0", torch.int8))
            with nvidia((7, 5)):
                self.assertEqual(supported("cuda:0", torch.float32), "native")
                self.assertIsNone(supported("cuda:0", torch.bfloat16))
                self.assertIsNone(supported("cuda:0", torch.float32, backend="triton"))

    def test_missing_software_gives_none(self):
        with amd("gfx1200"), mock.patch(
                "faiss.contrib.e_means.torch_backend._get_triton_assign_rows",
                side_effect=ImportError("no triton")):
            self.assertIsNone(supported("cuda:0", torch.bfloat16))
            self.assertIsNone(supported("cuda:0", torch.bfloat16, backend="triton"))


def _rows(dtype, device, n=512, d=64):
    x = torch.randn(n, d, generator=torch.Generator().manual_seed(0)).to(device)
    if dtype is torch.int8:
        return (x * 20).clamp(-127, 127).to(dtype)
    if dtype is torch.uint8:
        return (x * 20 + 128).clamp(0, 255).to(dtype)
    return x.to(dtype)


class SupportedMatchesTrainingTest(unittest.TestCase):

    def _check(self, device):
        for dtype in ALL:
            if device == "cuda" and dtype is FP8 and not hasattr(torch, "float8_e4m3fn"):
                continue
            x = _rows(dtype, device)
            for spherical in (False, True):
                for backend in BACKENDS:
                    with self.subTest(dtype=dtype, spherical=spherical, backend=backend):
                        model = QuantizedEmeans(64, 16, nepochs=1, spherical=spherical,
                                                assignment_backend=backend, quantizer=None)
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                model.train(x)
                            ran = model._resolved_assignment_backend
                        except Exception:
                            ran = None
                        self.assertEqual(
                            supported(device, dtype, spherical=spherical, backend=backend,
                                      d=64, k=16), ran)

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(GPU, "GPU required")
    def test_gpu(self):
        self._check("cuda")


class MemoryEstimateTest(unittest.TestCase):

    def test_cpu_is_zero_and_quantizer_is_refused(self):
        self.assertEqual(memory_estimate(Emeans(8, 4), 100, torch.float32, "cpu"), 0)
        model = QuantizedEmeans(8, 4, quantizer=CastQuantizer(8, torch.bfloat16))
        with self.assertRaises(ValueError):
            memory_estimate(model, 100, torch.float32, "cuda:0")
        with self.assertRaises(ValueError):
            memory_estimate(Emeans(8, 4), 0, torch.float32, "cuda:0")

    @unittest.skipUnless(GPU, "GPU required")
    def test_input_term(self):
        model = Emeans(64, 256)
        with_input = memory_estimate(model, 100_000, torch.float16, "cuda")
        without = memory_estimate(model, 100_000, torch.float16, "cuda", include_input=False)
        self.assertEqual(with_input - without, 100_000 * 64 * 2)

    @unittest.skipUnless(triton_gpu(), "Triton GPU required")
    def test_bound_holds_against_measured_peak(self):
        cases = [(Emeans, torch.float32), (Emeans, torch.bfloat16), (Emeans, torch.float16),
                 (QuantizedEmeans, torch.float32)]
        for n, d, k in ((20_000, 32, 64), (200_000, 64, 1024), (100_000, 64, 8192),
                        (100_000, 768, 512)):
            for cls, dtype in cases:
                for backend in ("triton", "native"):
                    for capture in (True, False):
                        with self.subTest(n=n, d=d, k=k, model=cls.__name__, dtype=dtype,
                                          backend=backend, capture=capture):
                            options = dict(nepochs=2, assignment_backend=backend,
                                           capture_assignments=capture)
                            if cls is QuantizedEmeans:
                                options["quantizer"] = None
                            estimate = memory_estimate(cls(d, k, **options), n, dtype, "cuda")
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            base = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            # Created in dtype, so no FP32 temporary counts.
                            x = torch.randn(n, d, device="cuda", dtype=dtype)
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                cls(d, k, **options).train(x)
                            torch.cuda.synchronize()
                            peak = torch.cuda.max_memory_allocated() - base
                            del x
                            self.assertGreaterEqual(estimate, peak)


class CheckFiniteTest(unittest.TestCase):

    def _check(self, device):
        for dtype in (torch.float32, torch.float16, torch.bfloat16, FP8):
            for bad in (float("nan"), float("inf")):
                if dtype is FP8 and bad == float("inf"):
                    continue  # E4M3FN has no infinity: inf becomes NaN, tested above
                x = torch.randn(300, 8, device=device).to(dtype)
                x[211, 5] = bad
                for make in (lambda **o: Emeans(8, 4, nepochs=1, **o),
                             lambda **o: QuantizedEmeans(8, 4, nepochs=1, quantizer=None, **o)):
                    with self.subTest(dtype=dtype, bad=bad):
                        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                            make(check_finite=True).train(x)
                        model = make(check_finite=True)
                        model.begin(300, torch.zeros(4, 8, device=device))
                        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
                            model.accumulate(x)
        clean = torch.randn(300, 8, device=device)
        Emeans(8, 4, nepochs=1, check_finite=True).train(clean)
        init = torch.zeros(4, 8, device=device)
        init[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "init_centroids"):
            Emeans(8, 4, nepochs=1, check_finite=True).train(clean, init)
        self.assertFalse(Emeans(8, 4).check_finite)

    def test_cpu(self):
        self._check("cpu")

    @unittest.skipUnless(GPU, "GPU required")
    def test_gpu(self):
        self._check("cuda")

    @unittest.skipUnless(triton_gpu(), "Triton GPU required")
    def test_bytes_are_accepted(self):
        q = QuantizedEmeans(64, 16, nepochs=1, quantizer=None, check_finite=True)
        q.train(_rows(torch.int8, "cuda"))


if __name__ == "__main__":
    unittest.main()
