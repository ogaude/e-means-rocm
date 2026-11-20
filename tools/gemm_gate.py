"""C gate: Triton BF16 assignment throughput against a vendor BLAS GEMM of the same shape.

    python tools/gemm_gate.py
    python tools/gemm_gate.py --rows 65536 --d 128 768 --k 4096 65536 --threshold 0.85

The question: is a hand-written AMD assignment kernel (the counterpart of the CuTe
backend on H100) worth building? The assignment does a (rows x d) @ (d x k) product
plus a row-wise argmin. A plain GEMM of that shape, without the argmin and with the
result written out, is what an unfused vendor kernel reaches; a fused hand-written
kernel could at best approach it. If Triton is within the threshold of the GEMM
(time ratio GEMM / Triton >= threshold on the median shape; default 0.80, i.e. Triton
reaches at least 80% of the GEMM's throughput),
the port is not worth its maintenance cost.

Per shape it reports, as milliseconds and TFLOP/s (2 * rows * k * d):
  triton     _launch_assignment (the real path: tuned table or upstream constants,
             split-K when the heuristic picks it), BF16 in, FP32 distances + labels out
  gemm       torch.mm, BF16 in, BF16 out, best of the BLAS libraries PyTorch offers
             (on ROCm: rocBLAS via "cublas", hipBLASLt via "cublaslt", Composable
             Kernel via "ck" where supported, e.g. gfx942)
  gemm32     torch.addmm(out_dtype=float32), BF16 in, FP32 out, same libraries
  unfused    gemm32 + norm add + argmin in PyTorch (what the native backend does)
"""
import argparse
import json
import statistics

import torch
import triton
import triton.testing

from faiss.contrib.e_means import triton_assignment as ta


def bench(fn):
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=50, rep=200)


def best_blas(fn):
    """Best time over the BLAS libraries PyTorch can select; returns (ms, library)."""
    out = []
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    # CK GEMM exists on CDNA only; elsewhere PyTorch warns and silently falls back.
    libs = ("cublas", "cublaslt", "ck") if arch.startswith(("gfx94", "gfx95")) else (
        "cublas", "cublaslt")
    for lib in libs:
        try:
            torch.backends.cuda.preferred_blas_library(lib)
            out.append((bench(fn), lib))
        except Exception:
            pass
    try:
        torch.backends.cuda.preferred_blas_library("default")
    except Exception:
        pass
    return min(out) if out else (float("nan"), "none")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int, default=[65536, 262144])
    ap.add_argument("--d", nargs="+", type=int, default=[64, 128, 256, 768])
    ap.add_argument("--k", nargs="+", type=int, default=[1024, 4096, 16384, 65536])
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--max-gib", type=float, default=48.0,
                    help="skip shapes whose FP32 GEMM output exceeds this")
    ap.add_argument("--out", default="gemm_gate.jsonl")
    a = ap.parse_args()
    props = torch.cuda.get_device_properties(0)
    print("device:", props.name, "| target:",
          triton.runtime.driver.active.get_current_target().arch, flush=True)
    print("%7s %4s %6s | %15s | %15s %-9s | %15s | %15s | %6s %6s" % (
        "rows", "d", "k", "triton ms TF/s", "gemm ms TF/s", "lib", "gemm32 ms TF/s",
        "unfused ms TF/s", "g/t", "g32/t"))
    torch.manual_seed(0)
    results = []
    with open(a.out, "w") as f:
        for d in a.d:
            for k in a.k:
                c = torch.randn(k, d, device="cuda").to(torch.bfloat16)
                norm = c.float().square().sum(1)
                for b in a.rows:
                    if b * k * 4 / 2**30 > a.max_gib:
                        print("%7d %4d %6d | skipped: output above --max-gib" % (b, d, k))
                        continue
                    x = torch.randn(b, d, device="cuda").to(torch.bfloat16)
                    dist = torch.empty(b, device="cuda")
                    labels = torch.empty(b, device="cuda", dtype=torch.int64)
                    out16 = torch.empty(b, k, device="cuda", dtype=torch.bfloat16)
                    out32 = torch.empty(b, k, device="cuda", dtype=torch.float32)

                    t_tri = bench(lambda: ta._launch_assignment(x, c, norm, dist, labels,
                                                                False))
                    t_gemm, lib = best_blas(lambda: torch.mm(x, c.T, out=out16))
                    t_g32, lib32 = best_blas(
                        lambda: torch.addmm(norm.expand(b, k), x, c.T, beta=1.0, alpha=-2.0,
                                            out_dtype=torch.float32, out=out32))

                    def unfused():
                        torch.addmm(norm.expand(b, k), x, c.T, beta=1.0, alpha=-2.0,
                                    out_dtype=torch.float32, out=out32)
                        return out32.argmin(1)
                    t_unf = bench(unfused)
                    # Sanity: Triton labels agree with the unfused reference (ties allowed).
                    ref = unfused()
                    agree = float((ref == labels).float().mean())

                    flop = 2.0 * b * k * d
                    tf = lambda ms: flop / (ms * 1e-3) / 1e12
                    r = dict(rows=b, d=d, k=k, triton_ms=t_tri, gemm_ms=t_gemm, gemm_lib=lib,
                             gemm32_ms=t_g32, gemm32_lib=lib32, unfused_ms=t_unf,
                             triton_tflops=tf(t_tri), gemm_tflops=tf(t_gemm),
                             gemm_over_triton=t_gemm / t_tri, gemm32_over_triton=t_g32 / t_tri,
                             label_agreement=agree)
                    results.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print("%7d %4d %6d | %7.3f %7.1f | %7.3f %7.1f %-9s | %7.3f %7.1f | "
                          "%7.3f %7.1f | %6.2f %6.2f%s" % (
                              b, d, k, t_tri, tf(t_tri), t_gemm, tf(t_gemm), lib, t_g32,
                              tf(t_g32), t_unf, tf(t_unf), t_gemm / t_tri, t_g32 / t_tri,
                              "" if agree > 0.99 else "  labels agree %.3f" % agree),
                          flush=True)
                    del out16, out32
                    torch.cuda.empty_cache()

    if not results:
        return
    ratios = [r["gemm_over_triton"] for r in results]
    below = [r for r in results if r["gemm_over_triton"] < a.threshold]
    print("\nGEMM/Triton time ratio (1.0 = Triton as fast as the bare GEMM): "
          "median %.2f, min %.2f, max %.2f over %d shapes" % (
              statistics.median(ratios), min(ratios), max(ratios), len(ratios)))
    print("Shapes where Triton is more than %.0f%% slower than the GEMM: %d" % (
        (1 / a.threshold - 1) * 100, len(below)))
    for r in below:
        print("   rows=%d d=%d k=%d  ratio %.2f" % (r["rows"], r["d"], r["k"],
                                                   r["gemm_over_triton"]))
    verdict = ("STOP: Triton is within the threshold; no hand-written backend"
               if statistics.median(ratios) >= a.threshold
               else "GAP: Triton trails the GEMM on the median shape; consider the FlyDSL backend")
    print("Verdict (median shape, threshold %.2f): %s" % (a.threshold, verdict))


if __name__ == "__main__":
    main()
