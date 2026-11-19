"""Sweep Triton assignment-kernel configurations on the current GPU.

Calls the real triton_assignment._assign_kernel with explicit tile sizes, warps and
stages, checks every configuration against the upstream one (same labels; distances
within FP32 rounding), and reports the time per assignment call.

    python tools/tune_assign.py                         # sweep default shapes and grid
    python tools/tune_assign.py --dtypes bf16 --d 64 128 --k 4096 --top 5
    python tools/tune_assign.py --compare               # upstream vs per-family table only

"upstream" is the configuration from the module constants (what NVIDIA uses);
"table" is what _launch_assignment picks on this GPU (_AMD_DIRECT_CONFIGS, or the
constants when the family has no entry). Times are medians from
triton.testing.do_bench (ms). Configurations that fail to compile (for example
LDS exhaustion) are counted as failed and skipped.
"""
import argparse
import itertools

import torch
import triton
import triton.testing

from faiss.contrib.e_means import triton_assignment as ta

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def table(dtype, d):
    """The direct-kernel configuration _launch_assignment picks on this GPU."""
    reduced = dtype != torch.float32
    tuned = ta._amd_direct_config(torch.device("cuda"), reduced, d)
    if tuned is None:
        return upstream(dtype, d)
    bb, bk, bd, w, s = tuned
    full = reduced and ta._DIRECT_REDUCED_BLOCK_D < d <= ta._FULL_D_MAX_D
    return dict(BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=max(16, triton.next_power_of_2(d)) if full else bd,
                FULL_D=full, num_warps=w, num_stages=s)


def upstream(dtype, d):
    """The direct-kernel configuration from the module constants (as on NVIDIA)."""
    reduced = dtype != torch.float32
    if reduced and ta._DIRECT_REDUCED_BLOCK_D < d <= ta._FULL_D_MAX_D:
        bb, bk, w, s = ta._FULL_D_CONFIG
        return dict(BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=max(16, triton.next_power_of_2(d)),
                    FULL_D=True, num_warps=w, num_stages=s)
    return dict(BLOCK_B=128 if reduced and d > ta._FULL_D_MAX_D and d % 16 == 0 else ta._DIRECT_BLOCK_B,
                BLOCK_K=ta._DIRECT_BLOCK_K if reduced else 128,
                BLOCK_D=ta._DIRECT_REDUCED_BLOCK_D if reduced else ta._DIRECT_FP32_BLOCK_D,
                FULL_D=False, num_warps=8, num_stages=ta._NUM_STAGES)


def candidates(dtype, d, quick):
    reduced = dtype != torch.float32
    full_d_ok = reduced and ta._DIRECT_REDUCED_BLOCK_D < d <= ta._FULL_D_MAX_D
    for bb, bk, w, s in itertools.product(
            (32, 64, 128), (64, 128, 256), (2, 4, 8) if not quick else (4, 8), (1, 2, 3)):
        if full_d_ok:
            yield dict(BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=max(16, triton.next_power_of_2(d)),
                       FULL_D=True, num_warps=w, num_stages=s)
        for bd in ((32, 64, 128) if reduced else (16, 32, 64)):
            if bd > max(16, triton.next_power_of_2(d)):
                continue
            yield dict(BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=bd, FULL_D=False, num_warps=w, num_stages=s)


def launch(x, c, norm, dist, labels, cfg):
    reduced = c.dtype != torch.float32
    ta._assign_kernel[(triton.cdiv(x.shape[0], cfg["BLOCK_B"]),)](
        x, c, norm, dist, labels, x.shape[0], c.shape[0], x.shape[1],
        x.stride(0), x.stride(1), c.stride(0), c.stride(1),
        BLOCK_B=cfg["BLOCK_B"], BLOCK_K=cfg["BLOCK_K"], BLOCK_D=cfg["BLOCK_D"],
        FULL_D=cfg["FULL_D"], REDUCED=reduced, FP16=c.dtype == torch.float16, FP8=False,
        WIDE_CENTROID=False, SPHERICAL=False,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def label(cfg):
    return "B%-3d K%-3d D%-3d %s w%d s%d" % (cfg["BLOCK_B"], cfg["BLOCK_K"], cfg["BLOCK_D"],
                                          "full" if cfg["FULL_D"] else "tile",
                                          cfg["num_warps"], cfg["num_stages"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp16", "fp32"])
    ap.add_argument("--d", nargs="+", type=int, default=[32, 64, 128, 256, 768])
    ap.add_argument("--k", nargs="+", type=int, default=[1024, 4096, 16384])
    ap.add_argument("--rows", type=int, default=65536)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="fewer warp counts")
    ap.add_argument("--compare", action="store_true",
                    help="time only the upstream and per-family table configurations")
    a = ap.parse_args()
    torch.manual_seed(0)
    print("device:", torch.cuda.get_device_name(0), "| target:",
          triton.runtime.driver.active.get_current_target().arch, flush=True)
    for name in a.dtypes:
        dtype = DTYPES[name]
        for d in a.d:
            for k in a.k:
                x = torch.randn(a.rows, d, device="cuda").to(dtype)
                c = torch.randn(k, d, device="cuda").to(dtype)
                norm = c.float().square().sum(1)
                dist = torch.empty(a.rows, device="cuda")
                labels = torch.empty(a.rows, device="cuda", dtype=torch.int64)
                ref = upstream(dtype, d)
                launch(x, c, norm, dist, labels, ref)
                ref_labels, ref_dist = labels.clone(), dist.clone()
                t_ref = triton.testing.do_bench(lambda: launch(x, c, norm, dist, labels, ref))
                if a.compare:
                    cfg = table(dtype, d)
                    launch(x, c, norm, dist, labels, cfg)
                    torch.cuda.synchronize()
                    mismatch = int((labels != ref_labels).sum())
                    ok = not mismatch or torch.allclose(dist, ref_dist, rtol=1e-5, atol=1e-4)
                    t = triton.testing.do_bench(lambda: launch(x, c, norm, dist, labels, cfg))
                    print("%-4s d=%-4d k=%-6d upstream %8.3f ms  table %8.3f ms  %5.2fx  %s%s" % (
                        name, d, k, t_ref, t, t_ref / t, label(cfg),
                        "" if ok else "  RESULTS DIFFER"), flush=True)
                    continue
                results, failed = [], 0
                seen = set()
                for cfg in candidates(dtype, d, a.quick):
                    key = tuple(sorted(cfg.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        launch(x, c, norm, dist, labels, cfg)
                        torch.cuda.synchronize()
                        mismatch = int((labels != ref_labels).sum())
                        # Different tile orders may break exact ties differently;
                        # a label change must not change the winning distance.
                        bad = mismatch and not torch.allclose(dist, ref_dist, rtol=1e-5, atol=1e-4)
                        if bad:
                            failed += 1
                            continue
                        t = triton.testing.do_bench(lambda: launch(x, c, norm, dist, labels, cfg))
                        results.append((t, cfg, mismatch))
                    except Exception:
                        failed += 1
                results.sort(key=lambda r: r[0])
                print("\n%s d=%d k=%d rows=%d  upstream %s: %.3f ms  (%d configs, %d failed)" % (
                    name, d, k, a.rows, label(ref), t_ref, len(results), failed), flush=True)
                for t, cfg, mismatch in results[:a.top]:
                    print("   %.3f ms  %.2fx  %s%s" % (t, t_ref / t, label(cfg),
                          "  (%d tie labels differ)" % mismatch if mismatch else ""), flush=True)


if __name__ == "__main__":
    main()
