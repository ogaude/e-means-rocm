# e-means-rocm: encoded rows' revival statistics through the fused Triton kernel.
#
# Upstream b14b3d69 (#5954) computes the [p y | d2 - d1] terms of decoded INT8/FP8 rows with
# PyTorch index_add_; e-means-rocm routes them through accumulate_removal(add_stats=False).
# These tests check that the fused path adds the same terms (up to FP32 summation order),
# never touches the [x | 1] sums, and is what training uses.
import types
import unittest

import torch

from e_means_gpu_conditions import triton_gpu

TRITON = triton_gpu()

if TRITON:
    from faiss.contrib.e_means import e_means_quantized as eq
    from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
    from faiss.contrib.e_means.e_means_torch import _accumulate_removal_rows
    from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
    from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer
    from faiss.contrib.e_means.torch_backend import TritonBackend


def _model(k, d, generator):
    axes = torch.randn(k, d, generator=generator, device="cuda")
    axes /= axes.norm(dim=1, keepdim=True)
    centroids = torch.randn(k, d, generator=generator, device="cuda")
    model = types.SimpleNamespace(
        _execution_backend=TritonBackend(4096), _axes=axes.contiguous(),
        _batch_removal=torch.zeros(k, d + 1, device="cuda"))
    model._readout = lambda: centroids
    return model


@unittest.skipUnless(TRITON, "Triton GPU required (NVIDIA SM80+ or AMD target family)")
class FusedEncodedRemovalTest(unittest.TestCase):

    def test_matches_index_add_reference(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        # (n, d, k): a long segment (> 256 rows of one label) exercises the pieces kernel.
        for n, d, k, long_segment in ((5000, 64, 40, False), (3000, 128, 3, True), (700, 1, 8, False)):
            for margin_kind in ("float", "int", "none"):
                with self.subTest(n=n, d=d, k=k, margin=margin_kind):
                    values = torch.randn(n, d, generator=g, device="cuda")
                    labels = torch.randint(0, k, (n,), generator=g, device="cuda")
                    if long_segment:
                        labels[: n // 2] = 1
                    if margin_kind == "float":
                        margin = torch.rand(n, generator=g, device="cuda") * 4
                        margin[:4] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0])
                    elif margin_kind == "int":
                        margin = torch.randint(0, 1000, (n,), generator=g, device="cuda")
                    else:
                        margin = None
                    fused, reference = _model(k, d, g), None
                    reference = types.SimpleNamespace(**vars(fused))
                    reference._batch_removal = fused._batch_removal.clone()
                    self.assertTrue(eq._fused_removal_rows(fused, values, labels, margin))
                    _accumulate_removal_rows(reference, values, labels,
                                             torch.zeros(n, device="cuda") if margin is None else margin)
                    torch.testing.assert_close(fused._batch_removal, reference._batch_removal,
                                               rtol=1e-5, atol=1e-3)

    def test_leaves_cluster_sums_alone(self):
        from faiss.contrib.e_means import triton_assignment as ta

        g = torch.Generator(device="cuda").manual_seed(1)
        n, d, k = 4000, 32, 16
        values = torch.randn(n, d, generator=g, device="cuda")
        labels = torch.randint(0, k, (n,), generator=g, device="cuda")
        stats = torch.full((k, d + 1), 7.0, device="cuda")
        removal = torch.zeros(k, d + 1, device="cuda")
        axes = torch.nn.functional.normalize(torch.randn(k, d, generator=g, device="cuda"), dim=1)
        zero = torch.zeros(n, device="cuda")
        self.assertTrue(ta.accumulate_removal(values, labels, stats, zero, zero + 1, axes,
                                              torch.zeros(k, d, device="cuda"), removal, add_stats=False))
        self.assertTrue(bool((stats == 7.0).all()), "add_stats=False wrote the [x | 1] sums")
        self.assertTrue(bool(removal.abs().sum() > 0))

    def test_training_uses_the_fused_path(self):
        x = torch.randn(20000, 64, device="cuda")
        for quantizer_class in (Int8Quantizer, Fp8Quantizer):
            if quantizer_class is Fp8Quantizer:
                from e_means_gpu_conditions import fp8_gpu
                if not fp8_gpu():
                    continue
            with self.subTest(quantizer=quantizer_class.__name__):
                q = quantizer_class(64)
                q.train(x)
                codes = q.compute_codes(x)
                calls = []
                original = eq._fused_removal_rows

                def spy(*args, **kwargs):
                    calls.append(original(*args, **kwargs))
                    return calls[-1]

                eq._fused_removal_rows = spy
                try:
                    QuantizedEmeans(64, 32, nepochs=2, assignment_backend="triton",
                                    quantizer=q).train(codes)
                finally:
                    eq._fused_removal_rows = original
                self.assertTrue(calls and all(calls), calls[:4])


if __name__ == "__main__":
    unittest.main()
