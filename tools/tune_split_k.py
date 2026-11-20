"""Measure where split-K assignment pays off on the current GPU.

    python tools/tune_split_k.py
    python tools/tune_split_k.py --rows 512 2048 --k 16384 65536 --d 128

Split-K splits the centroid range over several programs per row block and adds a
reduction launch. _launch_assignment uses it for reduced precision when the row grid
is small (_select_split_k: _use_split_k plus a per-family row-block cap on AMD), with
a partition count from _split_k_partitions. Both rest on occupancy constants fitted on
NVIDIA (_SPLIT_K_TARGET_PROGRAMS = 512).

For each shape this times _launch_assignment three ways: forced direct kernel, split-K
at each power-of-two partition count, and the current heuristic. Labels are checked
against the direct kernel (ties allowed when the distance agrees). Small row counts
are the interesting ones: assignment of a small final batch, or of few queries.
"""
import argparse

import torch
import triton
import triton.testing

from faiss.contrib.e_means import triton_assignment as ta

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", nargs="+", default=["bf16"])
    ap.add_argument("--rows", nargs="+", type=int, default=[256, 1024, 4096, 16384, 65536])
    ap.add_argument("--d", nargs="+", type=int, default=[64, 128, 256, 768])
    ap.add_argument("--k", nargs="+", type=int, default=[4096, 16384, 65536])
    a = ap.parse_args()
    props = torch.cuda.get_device_properties(0)
    print("device:", props.name, "| target:",
          triton.runtime.driver.active.get_current_target().arch,
          "| multi_processor_count:", props.multi_processor_count,
          "| _SPLIT_K_TARGET_PROGRAMS:", ta._SPLIT_K_TARGET_PROGRAMS, flush=True)
    print("%-4s %6s %5s %6s | %9s | %-24s | %-22s | %s" % (
        "dt", "rows", "d", "k", "direct ms", "best split-K (P, ms, x)", "heuristic (P, ms)",
        "verdict"))

    orig_use, orig_parts = ta._select_split_k, ta._split_k_partitions
    device = torch.device("cuda")
    torch.manual_seed(0)
    try:
        for name in a.dtypes:
            dtype = DTYPES[name]
            for d in a.d:
                for k in a.k:
                    c = torch.randn(k, d, device="cuda").to(dtype)
                    norm = c.float().square().sum(1)
                    for b in a.rows:
                        x = torch.randn(b, d, device="cuda").to(dtype)
                        dist = torch.empty(b, device="cuda")
                        labels = torch.empty(b, device="cuda", dtype=torch.int64)
                        run = lambda: ta._launch_assignment(x, c, norm, dist, labels, False)

                        # Current heuristic first (module functions untouched).
                        heur_split = orig_use(device, b, k, d, True)
                        heur_p = orig_parts(b, k) if heur_split else 1
                        t_heur = triton.testing.do_bench(run)

                        ta._select_split_k = lambda *args, **kw: False
                        run()
                        torch.cuda.synchronize()
                        ref_labels, ref_dist = labels.clone(), dist.clone()
                        t_direct = triton.testing.do_bench(run)

                        best = None
                        max_p = max(1, min(triton.cdiv(k, ta._SPLIT_K_BLOCK_K),
                                           ta._SPLIT_K_MAX_PARTITIONS))
                        p = 2
                        ta._select_split_k = lambda *args, **kw: True
                        while p <= max_p:
                            ta._split_k_partitions = lambda *args, p=p: p
                            try:
                                run()
                                torch.cuda.synchronize()
                                diff = int((labels != ref_labels).sum())
                                if diff and not torch.allclose(dist, ref_dist, rtol=1e-5,
                                                               atol=1e-4):
                                    print("   P=%d: RESULTS DIFFER (%d labels)" % (p, diff))
                                else:
                                    t = triton.testing.do_bench(run)
                                    if best is None or t < best[1]:
                                        best = (p, t)
                            except Exception as exc:
                                print("   P=%d failed: %s" % (p, str(exc)[:120]))
                            p *= 2
                        ta._select_split_k, ta._split_k_partitions = orig_use, orig_parts

                        if best and best[1] < 0.95 * t_direct:
                            verdict = "split-K P=%d" % best[0]
                        else:
                            verdict = "direct"
                        heur_ok = (verdict == "direct") == (not heur_split)
                        print("%-4s %6d %5d %6d | %9.3f | %-24s | %-22s | %s%s" % (
                            name, b, d, k, t_direct,
                            "P=%-3d %8.3f %5.2fx" % (best[0], best[1], t_direct / best[1])
                            if best else "-",
                            "%s P=%-3d %8.3f" % ("split" if heur_split else "direct",
                                                 heur_p, t_heur),
                            verdict, "" if heur_ok else "   <- heuristic disagrees"),
                            flush=True)
    finally:
        ta._select_split_k, ta._split_k_partitions = orig_use, orig_parts


if __name__ == "__main__":
    main()
