"""Sweep Triton assignment-kernel configurations on the current GPU.

Calls the real triton_assignment._assign_kernel with explicit tile sizes, warps and
stages, checks every configuration against the upstream one (same labels; distances
within FP32 rounding), and reports the time per assignment call.

    python tools/tune_assign.py                         # sweep default shapes and grid
    python tools/tune_assign.py --dtypes bf16 --d 64 128 --k 4096 --top 5
    python tools/tune_assign.py --compare               # upstream vs per-family table only
    python tools/tune_assign.py --dtypes int8           # INT8 kernel (_assign_int8_kernel)

"upstream" is the configuration from the module constants (what NVIDIA uses);
"table" is what _launch_assignment picks on this GPU (_AMD_DIRECT_CONFIGS, or the
constants when the family has no entry). Times are medians from
triton.testing.do_bench (ms). Configurations that fail to compile (for example
LDS exhaustion) are counted as failed and skipped.

INT8 ("--dtypes int8") sweeps _assign_int8_kernel instead. Its "upstream" is NVIDIA's
launch configuration with the stage count from _AMD_INT8_STAGES (upstream's three stages
do not fit outside gfx942); "table" is what _int8_launch_config picks on this GPU.
Distances are exact integers and ties go to the lowest index, so every candidate must
reproduce the reference distances and labels exactly. d must be a multiple of 16 (the
padded width the launcher would use anyway).
"""
import argparse
import itertools

import torch
import triton
import triton.testing

from faiss.contrib.e_means import triton_assignment as ta

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _family():
    from faiss.contrib.e_means.device_profile import device_profile

    return device_profile(torch.device("cuda")).family if torch.version.hip else None


def int8_cfg(full, bb, bk, bd, w, s):
    return dict(FULL_D=full, BLOCK_B=bb, BLOCK_K=bk, BLOCK_D=bd, num_warps=w, num_stages=s)


def int8_upstream(d):
    """NVIDIA's INT8 launch configuration, with the AMD family's stage count."""
    return int8_cfg(*ta._int8_default_config(d, _family()))


def int8_table(d):
    return int8_cfg(*ta._int8_launch_config(d, _family()))


def int8_candidates(d, quick):
    for full in ((True, False) if d <= 256 else (False,)):
        for bb, bk, w, s in itertools.product(
                (32, 64, 128), (64, 128, 256), (2, 4, 8) if not quick else (4, 8), (1, 2, 3)):
            if full:
                yield int8_cfg(True, bb, bk, max(16, triton.next_power_of_2(d)), w, s)
                continue
            for bd in (32, 64, 128, 256):
                if bd <= max(16, triton.next_power_of_2(d)):
                    yield int8_cfg(False, bb, bk, bd, w, s)


def int8_launch(x, c, norm, dist, labels, cfg):
    d, bk = x.shape[1], cfg["BLOCK_K"]
    ta._assign_int8_kernel[(triton.cdiv(x.shape[0], cfg["BLOCK_B"]),)](
        x, c, norm, dist, labels, x.shape[0], c.shape[0], d,
        x.stride(0), x.stride(1), c.stride(0), c.stride(1),
        UNSIGNED=False, BLOCK_B=cfg["BLOCK_B"], BLOCK_K=bk, BLOCK_D=cfg["BLOCK_D"],
        WIDE_CENTROID=False, FULL_D=cfg["FULL_D"],
        PACKED=48896 * d * bk + bk - 1 < 2**31,
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def int8_data(rows, d, k):
    if d % ta._INT8_D_ALIGNMENT:
        raise SystemExit("INT8: d must be a multiple of %d" % ta._INT8_D_ALIGNMENT)
    ta._load_int8_libdevice()
    x = torch.randint(-128, 128, (rows, d), device="cuda", dtype=torch.int8)
    c, norm = ta.prepare_int8_centroids(torch.randn(k, d, device="cuda") * 40, "int8")
    dist = torch.empty(rows, device="cuda", dtype=torch.int64)
    labels = torch.empty(rows, device="cuda", dtype=torch.int64)
    return x, c, norm, dist, labels


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
    """Short text for a configuration."""
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
        int8 = name == "int8"
        dtype = DTYPES.get(name)
        for d in a.d:
            for k in a.k:
                if int8:
                    x, c, norm, dist, labels = int8_data(a.rows, d, k)
                    run, ref = int8_launch, int8_upstream(d)
                else:
                    x = torch.randn(a.rows, d, device="cuda").to(dtype)
                    c = torch.randn(k, d, device="cuda").to(dtype)
                    norm = c.float().square().sum(1)
                    dist = torch.empty(a.rows, device="cuda")
                    labels = torch.empty(a.rows, device="cuda", dtype=torch.int64)
                    run, ref = launch, upstream(dtype, d)
                run(x, c, norm, dist, labels, ref)
                ref_labels, ref_dist = labels.clone(), dist.clone()
                t_ref = triton.testing.do_bench(lambda: run(x, c, norm, dist, labels, ref))
                if a.compare:
                    cfg = int8_table(d) if int8 else table(dtype, d)
                    run(x, c, norm, dist, labels, cfg)
                    torch.cuda.synchronize()
                    mismatch = int((labels != ref_labels).sum())
                    if int8:
                        ok = not mismatch and torch.equal(dist, ref_dist)
                    else:
                        ok = not mismatch or torch.allclose(dist, ref_dist, rtol=1e-5, atol=1e-4)
                    t = triton.testing.do_bench(lambda: run(x, c, norm, dist, labels, cfg))
                    print("%-4s d=%-4d k=%-6d upstream %8.3f ms  table %8.3f ms  %5.2fx  %s%s" % (
                        name, d, k, t_ref, t, t_ref / t, label(cfg),
                        "" if ok else "  RESULTS DIFFER"), flush=True)
                    continue
                results, failed = [], 0
                seen = set()
                for cfg in (int8_candidates(d, a.quick) if int8
                            else candidates(dtype, d, a.quick)):
                    key = tuple(sorted(cfg.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        run(x, c, norm, dist, labels, cfg)
                        torch.cuda.synchronize()
                        mismatch = int((labels != ref_labels).sum())
                        if int8:  # exact integers, lowest-index ties: must be identical
                            bad = mismatch or not torch.equal(dist, ref_dist)
                        else:
                            # Different tile orders may break exact ties differently;
                            # a label change must not change the winning distance.
                            bad = mismatch and not torch.allclose(dist, ref_dist, rtol=1e-5,
                                                                  atol=1e-4)
                        if bad:
                            failed += 1
                            continue
                        t = triton.testing.do_bench(lambda: run(x, c, norm, dist, labels, cfg))
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
