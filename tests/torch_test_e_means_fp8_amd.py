# FP8 (E4M3) numerics on AMD GPUs with a native E4M3 matrix instruction (gfx950,
# gfx120x). Runs on any Triton GPU that supports FP8 assignment; skipped elsewhere.
#
# What this guards:
#   * the Triton E4M3 encode (_cast_fp8_centroids_kernel, value.to(float8e4nv,
#     rtne)) gives exactly PyTorch's float8_e4m3fn codes for every value the range
#     check admits (|v| <= 448): the whole FP16 grid, every tie between neighbouring
#     codes and its FP32 neighbours, subnormals, and random FP32;
#   * the same with a quantizer: the scaled (value * scale) and affine (one-rounding
#     fma(value, scale, shift)) paths of the cast kernel against PyTorch, and the peak;
#   * _pow2, which replaces libdevice.ldexp on AMD in the scaled FP8 split, gives
#     exactly torch.ldexp for every finite E4M3 value and every decode exponent in
#     [-127, 127], subnormal results included.
#   * opt-in (EMEANS_EXHAUSTIVE_TESTS=1): every FP32 value with |v| <= 448, 2.28e9 of
#     them, through the cast kernel (a few seconds on an RX 9060 XT).

import os
import unittest
from types import SimpleNamespace

import torch

from e_means_gpu_conditions import amd_fp8_gpu, nvidia_sm80

try:
    import triton
    import triton.language as tl
    from faiss.contrib.e_means import triton_assignment as ta
    HAVE_TRITON = True
except Exception:  # pragma: no cover - environment without triton
    HAVE_TRITON = False

FP8_GPU = HAVE_TRITON and (amd_fp8_gpu() or (
    nvidia_sm80() and torch.cuda.get_device_capability() in ((9, 0), (10, 0), (10, 3))))
E4M3_MAX = 448.0


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t.view(torch.uint8)


def _triton_codes(values):
    """E4M3 codes from the production cast kernel (no quantizer: plain encode)."""
    source = values.reshape(-1, 1).contiguous()
    peak = torch.zeros((), device=values.device)
    codes, _ = ta._cast_fp8_centroids(source, None, peak)
    return codes[:, 0].contiguous()


if HAVE_TRITON:
    @triton.jit
    def _scale_kernel(values, exponents, out, n, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = lane < n
        value = tl.load(values + lane, mask=mask, other=0.0)
        exponent = tl.load(exponents + lane, mask=mask, other=0)
        tl.store(out + lane, value * ta._pow2(exponent), mask=mask)


@unittest.skipUnless(FP8_GPU, "FP8-capable Triton GPU required")
class Fp8EncodeParityTest(unittest.TestCase):

    def assert_same_codes(self, values):
        values = values.float().cuda()
        values = values[values.abs() <= E4M3_MAX]
        expected = values.to(torch.float8_e4m3fn)
        actual = _triton_codes(values)
        mismatch = _bits(actual) != _bits(expected)
        if bool(mismatch.any()):
            bad = values[mismatch][:8]
            self.fail("E4M3 codes differ for %d values, e.g. %s -> triton %s, torch %s" % (
                int(mismatch.sum()), bad.tolist(),
                _bits(actual[mismatch][:8]).tolist(), _bits(expected[mismatch][:8]).tolist()))

    def test_whole_fp16_grid(self):
        grid = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        self.assert_same_codes(grid[torch.isfinite(grid)])

    def test_ties_and_their_neighbours(self):
        # Every finite non-negative E4M3 value, and the midpoint to the next one.
        codes = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        mid = (codes[:-1].double() + codes[1:].double()) / 2  # exact in FP32
        mid = mid.float()
        bits = mid.view(torch.int32)
        near = torch.cat([(bits + delta).view(torch.float32) for delta in (-2, -1, 0, 1, 2)])
        values = torch.cat([codes, near])
        self.assert_same_codes(torch.cat([values, -values]))

    def test_subnormals_and_tiny_values(self):
        tiny = torch.cat([
            torch.linspace(0, 2.0 ** -6, 100001),             # E4M3 subnormal range
            torch.tensor([2.0 ** -10, 2.0 ** -11, 2.0 ** -126, 1e-45, 0.0]),
        ])
        self.assert_same_codes(torch.cat([tiny, -tiny]))

    def test_random_fp32(self):
        g = torch.Generator().manual_seed(0)
        bits = torch.randint(-2**31, 2**31 - 1, (4_000_000,), generator=g, dtype=torch.int64)
        values = bits.to(torch.int32).view(torch.float32)
        values = values[torch.isfinite(values)]
        uniform = (torch.rand(4_000_000, generator=g) * 2 - 1) * E4M3_MAX
        self.assert_same_codes(torch.cat([values, uniform]))


@unittest.skipUnless(FP8_GPU, "FP8-capable Triton GPU required")
class Fp8QuantizedCastParityTest(unittest.TestCase):
    """The cast kernel's scaled and affine paths against PyTorch, codes and peak."""

    def _check(self, values, scale, bias):
        d = values.shape[1]
        reference = values * scale if bias is None else values * scale + bias
        keep = (reference.abs() <= E4M3_MAX).all(dim=1)
        values, reference = values[keep].contiguous(), reference[keep]
        quantizer = SimpleNamespace(scale=scale, _bias=bias)
        peak = torch.zeros((), device="cuda")
        codes, squares = ta._cast_fp8_centroids(values, quantizer, peak)
        expected = reference.to(torch.float8_e4m3fn)
        got = codes[:, :d].contiguous()
        mismatch = _bits(got) != _bits(expected)
        self.assertFalse(bool(mismatch.any()), "%d codes differ, e.g. %s" % (
            int(mismatch.sum()), reference[mismatch][:5].tolist()))
        self.assertEqual(float(peak), float(reference.abs().max()))
        torch.testing.assert_close(squares, expected.float().square().sum(dim=1),
                                   rtol=0, atol=0)

    def _rows(self, seed, n=200_000, d=64):
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows = torch.randn((n, d), device="cuda", generator=g) * 30
        # Exact E4M3 values and midpoints after scaling by 1/8: ties land on codes.
        codes = torch.arange(0, 0x7F, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        mids = (codes[:-1] + codes[1:]) / 2
        special = torch.cat([codes, mids, -codes, -mids]).cuda() * 8.0
        rows.view(-1)[: len(special)] = special
        return rows

    def test_scaled(self):
        for scale in (0.125, 1.0, 4.0):
            with self.subTest(scale=scale):
                self._check(self._rows(1), scale, None)

    def test_affine(self):
        for scale in (0.125, 2.0):
            with self.subTest(scale=scale):
                g = torch.Generator(device="cuda").manual_seed(2)
                bias = torch.randn(64, device="cuda", generator=g) * 20
                self._check(self._rows(3), scale, bias.contiguous())


@unittest.skipUnless(FP8_GPU and os.environ.get("EMEANS_EXHAUSTIVE_TESTS") == "1",
                     "set EMEANS_EXHAUSTIVE_TESTS=1 on an FP8-capable Triton GPU")
class Fp8ExhaustiveEncodeTest(unittest.TestCase):
    """Every FP32 value with |v| <= 448 encodes as torch.float8_e4m3fn does."""

    def test_every_value_up_to_448(self):
        limit = torch.tensor(E4M3_MAX).view(torch.int32).item()  # bits of 448.0
        chunk, checked, mismatches = 1 << 25, 0, 0
        for sign in (0, 1 << 31):
            for start in range(0, limit + 1, chunk):
                bits = torch.arange(start, min(start + chunk, limit + 1), device="cuda",
                                    dtype=torch.int64) | sign
                values = bits.to(torch.int32).view(torch.float32)
                got = _triton_codes(values)
                expected = values.to(torch.float8_e4m3fn)
                mismatches += int((_bits(got) != _bits(expected)).sum())
                checked += values.numel()
        self.assertEqual(checked, 2 * (limit + 1))
        self.assertEqual(mismatches, 0)


@unittest.skipUnless(FP8_GPU, "FP8-capable Triton GPU required")
class Pow2MatchesLdexpTest(unittest.TestCase):

    def test_every_e4m3_value_and_decode_exponent(self):
        codes = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        codes = codes[torch.isfinite(codes)].cuda()
        exponents = torch.arange(-127, 128, dtype=torch.int32, device="cuda")
        values = codes.repeat(len(exponents))
        exps = exponents.repeat_interleave(len(codes))
        out = torch.empty_like(values)
        _scale_kernel[(triton.cdiv(len(values), 1024),)](values, exps, out, len(values),
                                                        BLOCK=1024)
        expected = torch.ldexp(values, exps)
        same = (_bits(out) == _bits(expected)) | (torch.isnan(out) & torch.isnan(expected))
        if not bool(same.all()):
            i = (~same).nonzero()[:5, 0]
            self.fail("pow2 differs from ldexp: %s" % list(zip(
                values[i].tolist(), exps[i].tolist(), out[i].tolist(), expected[i].tolist())))


if __name__ == "__main__":
    unittest.main()
