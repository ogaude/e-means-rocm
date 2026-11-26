# Bit-level checks of the Triton helper kernels that replaced PTX inline assembly (W3)
# against independent references, on any GPU where the Triton backend runs (NVIDIA
# SM80+ or an AMD target family). These are the W9 hardware checks: they need a GPU.
#
# * EMA update: triton_assignment.ema_update == TorchBackend.update, bit for bit.
# * Centroid split (FP32 and BF16): _write_float_splits == TorchBackend.split.
# * The BF16 rounding helper == torch .to(torch.bfloat16) on the same device.
# * tl.math.div_rn == correctly rounded FP32 division (CPU reference).
# Inputs include zeros, signed zeros, subnormals (odd ones too), large finite values,
# infinities and exact ties.

import unittest

import torch

from e_means_gpu_conditions import triton_gpu

TRITON_GPU = triton_gpu()

if TRITON_GPU:
    import triton
    import triton.language as tl

    from faiss.contrib.e_means import triton_assignment as ta
    from faiss.contrib.e_means.torch_backend import TorchBackend

    @triton.jit
    def _bf16_bits_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        x = tl.load(x_ptr + i, mask=m, other=0.0)
        low, _ = ta._rn_bf16_bits(x, x)
        tl.store(out_ptr + i, low.to(tl.int32), mask=m)

    @triton.jit
    def _div_rn_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        a = tl.load(a_ptr + i, mask=m, other=1.0)
        b = tl.load(b_ptr + i, mask=m, other=1.0)
        tl.store(out_ptr + i, tl.math.div_rn(a, b), mask=m)


TINY = 2.0 ** -149            # smallest positive subnormal (odd mantissa)
BIG = 3.0e38                  # large finite, near FLT_MAX


def _special_values(device):
    return torch.tensor([0.0, -0.0, TINY, -TINY, 3 * TINY, 2.0 ** -126, 1.0, -1.0, 1.5,
                         1e-38, 1e30, -1e30, BIG, -BIG, float("inf"), float("-inf")],
                        device=device)


def _bits_equal(test, actual, expected, label):
    """Bitwise equality; NaNs only need to be NaN in both (payloads may differ)."""
    nan = torch.isnan(expected)
    test.assertTrue(torch.equal(torch.isnan(actual), nan), "%s: NaN positions differ" % label)
    a = actual[~nan].contiguous().view(torch.int32)
    e = expected[~nan].contiguous().view(torch.int32)
    bad = (a != e).nonzero()
    test.assertEqual(len(bad), 0, "%s: %d values differ, first at %s: %r vs %r" % (
        label, len(bad), bad[:1].tolist(),
        actual[~nan][bad[0]].item() if len(bad) else None,
        expected[~nan][bad[0]].item() if len(bad) else None))


@unittest.skipUnless(TRITON_GPU, "Triton GPU required (NVIDIA SM80+ or AMD target family)")
class TritonNumericsTest(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda")
        self.g = torch.Generator(device=self.device).manual_seed(20261119)

    def _ema_state(self, k, d):
        g, dev = self.g, self.device
        assert k >= 6
        centroids = torch.randn(k, d, generator=g, device=dev) * 4
        centroids[0] = 0.0
        width = min(16, d)
        centroids[1, :width] = _special_values(dev)[:width].clamp(-1e30, 1e30)
        usage = torch.rand(k, generator=g, device=dev) * 10
        usage[:6] = torch.tensor([0.0, TINY, 3 * TINY, 2.0 ** -126, 2.0 ** -16, 1e30], device=dev)
        counts = torch.randint(0, 50, (k,), generator=g, device=dev).float()
        counts[::7] = 0.0                       # untouched rows keep their state
        sums = torch.randn(k, d, generator=g, device=dev) * counts[:, None]
        sums[2] = TINY                          # subnormal means
        sums[3] = 3 * TINY
        counts[2:4] = 1.0
        return centroids, usage, sums, counts

    def test_ema_update_matches_torch(self):
        cases = [(0.5, 0.5, False), (0.125, 0.875, False), (1.0, 0.0, False),
                 (0.3, 0.7, True), (2.0 ** -20, 1 - 2.0 ** -20, False)]
        for k, d in ((257, 129), (64, 1024), (7, 1)):
            for alpha, decay, replace in cases:
                for device_rates in (False, True):
                    for count_dtype in (torch.float32, torch.int64):
                        with self.subTest(k=k, d=d, alpha=alpha, replace=replace,
                                          device_rates=device_rates, counts=count_dtype):
                            c, u, sums, counts = self._ema_state(k, d)
                            counts = counts.to(count_dtype)
                            c_ref, u_ref = c.clone(), u.clone()
                            a = torch.tensor(alpha, device=self.device) if device_rates else alpha
                            b = torch.tensor(decay, device=self.device) if device_rates else decay
                            ran = ta.ema_update(sums, counts, c, u, a, b, replace)
                            self.assertTrue(ran, "Triton EMA kernel fell back to PyTorch")
                            TorchBackend(4096).update(c_ref, u_ref, None, sums, counts,
                                                      alpha, decay, replace)
                            torch.cuda.synchronize()
                            _bits_equal(self, c, c_ref, "centroids")
                            _bits_equal(self, u, u_ref, "usage")

    def _split_state(self, k, d):
        g, dev = self.g, self.device
        centroids = torch.randn(k, d, generator=g, device=dev)
        centroids[0] = 0.0                                  # collapsed donor
        centroids[1] = TINY                                 # subnormal donor
        width = min(16, d)
        centroids[2, :width] = _special_values(dev)[:width].clamp(-BIG, BIG)
        centroids[3] = 1.0                                  # BF16 collapse at 1 +- eps
        usage = torch.rand(k, generator=g, device=dev) * 8
        usage[:5] = torch.tensor([TINY, 3 * TINY, 0.0, 2.0 ** -126, 1e30], device=dev)
        source = torch.arange(0, k // 2, device=dev, dtype=torch.int64)
        target = torch.arange(k // 2, k // 2 * 2, device=dev, dtype=torch.int64)
        return centroids, usage, source, target

    def _split_axes(self, k, d):
        # Unit axes and FP64 |U| as cost-based revival keeps them, plus the edge cases the
        # kernel must treat as "no offset" (zero, infinite and NaN norms).
        dev = self.device
        axes = torch.randn(k, d, generator=self.g, device=dev)
        axes /= axes.norm(dim=1, keepdim=True)
        norm = (torch.rand(k, generator=self.g, device=dev).double() * 50) ** 2
        norm[: min(k, 4)] = torch.tensor([0.0, float("inf"), float("nan"), 1e-300],
                                         device=dev, dtype=torch.float64)[: min(k, 4)]
        return axes.contiguous(), norm.contiguous()

    def test_split_matches_torch(self):
        # Upstream b14b3d69 replaced the epsilon split by c -/+ s u along the cluster axis
        # (cost-based revival); its FP64 step was PTX, now plain FP64 Triton operations.
        for dtype in (None, torch.float32, torch.bfloat16):
            for k, d in ((64, 129), (16, 1024), (8, 1)):
                for with_axes in (False, True):
                    with self.subTest(dtype=dtype, k=k, d=d, axes=with_axes):
                        c, u, source, target = self._split_state(k, d)
                        axes, norm = self._split_axes(k, d) if with_axes else (None, None)
                        c_ref, u_ref = c.clone(), u.clone()
                        ran = ta._write_float_splits(c, u, source, target, dtype, axes, norm)
                        self.assertTrue(ran, "Triton split kernel fell back to PyTorch")
                        TorchBackend(4096).split(c_ref, u_ref, source, target, dtype, axes, norm)
                        torch.cuda.synchronize()
                        if not with_axes:
                            # Upstream: without axes the kernel keeps the donor value
                            # (low = high = c), while TorchBackend.split computes c + 0.0,
                            # which turns a -0.0 donor into +0.0 (on NVIDIA as well). Only the
                            # sign of a zero may differ; it changes no distance.
                            zero = (c == 0) & (c_ref == 0)
                            c, c_ref = c.masked_fill(zero, 0.0), c_ref.masked_fill(zero, 0.0)
                        _bits_equal(self, c, c_ref, "centroids")
                        _bits_equal(self, u, u_ref, "usage")

    def test_bf16_rounding_matches_torch(self):
        dev = self.device
        random_bits = torch.randint(-2 ** 31, 2 ** 31 - 1, (1 << 20,), generator=self.g,
                                    device=dev, dtype=torch.int64).to(torch.int32)
        # Exact ties: the dropped 16 bits are 0x8000, with even and odd kept LSB.
        ties = (torch.arange(0, 1 << 16, device=dev, dtype=torch.int64) << 16) | 0x8000
        ties = torch.where(ties >= 2 ** 31, ties - 2 ** 32, ties).to(torch.int32)
        x = torch.cat([random_bits.view(torch.float32), ties.view(torch.float32),
                       _special_values(dev)])
        out = torch.empty(x.numel(), dtype=torch.int32, device=dev)
        _bf16_bits_kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024,
                                                           enable_fp_fusion=False)
        expected = x.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
        nan = torch.isnan(x)
        self.assertTrue(bool(((out[nan] & 0x7F80) == 0x7F80).all()
                             and ((out[nan] & 0x7F) != 0).all()), "NaN not kept as NaN")
        bad = (out[~nan] != expected[~nan]).nonzero()
        self.assertEqual(len(bad), 0, "BF16 bits differ at %s" % bad[:3].tolist())

    def test_div_rn_is_correctly_rounded(self):
        dev = self.device
        n = 1 << 20
        a = torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), generator=self.g, device=dev,
                          dtype=torch.int64).to(torch.int32).view(torch.float32)
        b = torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), generator=self.g, device=dev,
                          dtype=torch.int64).to(torch.int32).view(torch.float32)
        s = _special_values(dev)
        a = torch.cat([a, s.repeat_interleave(len(s)), torch.full((64,), TINY, device=dev)])
        b = torch.cat([b, s.repeat(len(s)), torch.arange(1, 65, device=dev).float()])
        out = torch.empty_like(a)
        _div_rn_kernel[(triton.cdiv(a.numel(), 1024),)](a, b, out, a.numel(), BLOCK=1024,
                                                        enable_fp_fusion=False)
        expected = (a.cpu() / b.cpu()).to(dev)   # x86 FP32 division is correctly rounded
        _bits_equal(self, out, expected, "div_rn")


if __name__ == "__main__":
    unittest.main()
