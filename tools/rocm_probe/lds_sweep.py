"""Offline LDS sweep for the Triton kernels that upstream's b14b3d69 sync added.

No GPU needed: each kernel is compiled for the four AMD targets at the block sizes its
launcher can pick (smallest to largest d), and the compiled kernel's shared memory (LDS) is
compared with the family's budget from device_profile. The assignment kernels have their
own check in tests/torch_test_e_means_launch_config.py.

    ROCR_VISIBLE_DEVICES= nice -n 19 python tools/rocm_probe/lds_sweep.py [--arch gfx1200]

Types mirror the launch sites (triton_assignment.py, ivf_fine.py). Exit status 1 if a
configuration is over budget or fails to compile.
"""
import argparse
import itertools
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from faiss.contrib.e_means import device_profile as dp
from faiss.contrib.e_means import ivf_fine as ivf
from faiss.contrib.e_means import ivf_fine_triton as ift
from faiss.contrib.e_means import triton_assignment as ta

ARCHS = ("gfx942", "gfx950", "gfx1100", "gfx1200")
P32, P64, I64P, I32P, I16P = "*fp32", "*fp64", "*i64", "*i32", "*i16"


def np2(x):
    return triton.next_power_of_2(x)


def segment_config(d, long_batch):
    """_segment_sums_kernel: (BLOCK_R, warps) as segment_sums picks them."""
    bd = np2(d)
    if long_batch:
        return 16, 4
    return (4, 1) if bd <= 128 else (2, 2 if bd <= 512 else 4)


def removal_config(d):
    """accumulate_removal: (BLOCK_R, warps)."""
    bd = np2(d)
    return (4, 1) if bd <= 128 else (2, 2 if bd <= 512 else 4 if bd <= 2048 else 8)


def cases():
    """(kernel name, fn, signature, constexprs, num_warps, num_stages, label)."""
    widths = (16, 64, 128, 512, 2048, 4096)
    # Closing: block = min(128, np2(d)); warps 1/2/4.
    for d, cast in itertools.product((16, 64, 128, 4096), (0, 1, 2)):
        block = min(128, np2(d))
        warps = 1 if block <= 32 else 2 if block <= 64 else 4
        yield ("_closing_partials_kernel", ta._closing_partials_kernel,
               dict(x=P32, stride_x="i64", stride_inner="i64", index=I64P, starts=I64P, offsets=I64P,
                    owner=I64P, left=P64, right=P64, k="i32", step="i64", d="i32"),
               dict(PIECE=ta._CLOSING_PIECE, BLOCK_D=block, UNROLL=8, CAST=cast, UNIT_INNER=True),
               warps, 1, "d=%d cast=%d" % (d, cast))
        if cast == 0:
            yield ("_closing_flush_kernel", ta._closing_flush_kernel,
                   dict(starts=I64P, offsets=I64P, left=P64, right=P64, sums=P32, stride_s="i64",
                        counts=P64, step="i64", d="i32"),
                   dict(PIECE=ta._CLOSING_PIECE, BLOCK_D=block), warps, 1, "d=%d" % d)
    # Removal statistics (cost-based revival), d <= 4096; the EMA fold only up to 2048.
    for d, ema, scalars in itertools.product(widths, (False, True), (False, True)):
        if ema and np2(d) > 2048:
            continue
        br, warps = removal_config(d)
        scal = P32 if scalars else "fp32"
        common = dict(BLOCK_R=br, BLOCK_D=np2(d), EMA=ema, SCALARS=scalars, LONG=ta._LONG_SEGMENT)
        yield ("_removal_stats_kernel", ta._removal_stats_kernel,
               dict(rows=P32, order=I64P, offsets=I16P, distance=P32, second=P32, axes=P32,
                    centroids=P32, stats=P32, removal=P32, d="i32", row_stride="i64",
                    stats_stride="i64", removal_stride="i64", alpha=scal, decay=scal),
               common, warps, 1, "d=%d ema=%d scalars=%d" % (d, ema, scalars))
        yield ("_long_pieces_kernel", ta._long_pieces_kernel,
               dict(rows=P32, order=I64P, offsets=I16P, sorted_labels=I16P, partial=P32, counter=I32P,
                    distance=P32, second=P32, axes=P32, centroids=P32, stats=P32, removal=P32,
                    d="i32", n="i32", row_stride="i64", stats_stride="i64", removal_stride="i64",
                    alpha=scal, decay=scal),
               common, warps, 1, "d=%d ema=%d scalars=%d" % (d, ema, scalars))
    # Segment sums (FP32/BF16 rows, d <= 4096).
    for d, dtype, long_batch in itertools.product(widths, ("fp32", "bf16"), (False, True)):
        br, warps = segment_config(d, long_batch)
        yield ("_segment_sums_kernel", ta._segment_sums_kernel,
               dict(rows="*" + dtype, order=I64P, offsets=I16P, sorted_labels=I16P, partial=P32,
                    counter=I32P, stats=P32, d="i32", n="i32", pieces="i32", row_stride="i64",
                    stats_stride="i64"),
               dict(BLOCK_R=br, BLOCK_D=np2(d), LONG=ta._LONG_SEGMENT), warps, 1,
               "d=%d %s %s" % (d, dtype, "rows>=8k" if long_batch else "short"))
    # Removal update and split: widths up to d + 1 = 4097.
    for d, scalars in itertools.product(widths, (False, True)):
        bw = np2(d + 1)
        scal = P32 if scalars else "fp32"
        yield ("_removal_update_kernel", ta._removal_update_kernel,
               dict(stats=P32, batch=P32, axes=P32, alpha=scal, decay=scal, k="i32", d="i32",
                    stats_stride="i64", batch_stride="i64"),
               dict(ROWS=max(1, 2048 // bw), BLOCK_W=bw, SCALARS=scalars, axis_norms=None), 4, 1,
               "d=%d scalars=%d" % (d, scalars))
        if not scalars:
            yield ("_removal_split_kernel", ta._removal_split_kernel,
                   dict(stats=P32, axes=P32, source=I64P, target=I64P, width="i32", d="i32",
                        stats_stride="i64"),
                   dict(BLOCK_W=bw, BLOCK_D=np2(d), count=None), 4, 1, "d=%d" % d)
    # Deferred second-label resolve: TILE = the direct kernel's BLOCK_K.
    for d, tile, rows, dtype in itertools.product((16, 64, 128), (32, 64, 128, 256), (1, ta._RESOLVE_ROWS),
                                                  ("bf16", "fp16", "fp32")):
        yield ("_resolve_second_kernel", ta._resolve_second_kernel,
               dict(x_ptr="*" + dtype, centroid_ptr="*" + dtype, centroid_norm_ptr=P32, label_ptr=I64P,
                    second_label_ptr=I32P, pending_ptr=I32P, B="i32", K="i32", D="i32",
                    stride_xb="i64", stride_xd="i64", stride_ck="i64", stride_cd="i64"),
               dict(TILE=tile, BLOCK_D=min(128, np2(d)), ROWS=rows, FP16=dtype == "fp16"), 4, 1,
               "d=%d tile=%d rows=%d %s" % (d, tile, rows, dtype))
    # Twin reassignment in closing: BLOCK_R 16, BLOCK_D <= 128.
    for d, cast in itertools.product((16, 128, 4096), (0, 1, 2)):
        yield ("_twin_route_kernel", ta._twin_route_kernel,
               dict(x=P32, stride_x="i64", stride_inner="i64", index=I64P, row_index=I64P, labels=I64P,
                    stale="*i1", key=I64P, target=I64P, source=I64P, born=I64P, moved_at=I64P,
                    centroids=P32, stride_c="i64", n="i32", begin="i64", span="i64", n_events="i32",
                    segment=I64P, d="i32"),
               dict(BLOCK_R=16, BLOCK_D=min(128, np2(d)), CAST=cast, LOCAL_ROWS=False, UNIT_INNER=True),
               4, 1, "d=%d cast=%d" % (d, cast))
    # IVF fine search (BF16 and FP32 lists): the tiles and HIP stage count of ivf_fine._search_triton.
    for ieee in (True, False):
        bm, bn, bk, _ = (64, 64, 32, 3) if ieee else (64, 128, 64, 4)
        dt = "fp32" if ieee else "bf16"
        yield ("_search_kernel", ift._search_kernel,
               dict(x16="*" + dt, vecs="*" + dt, norms=P32, ids=I64P, lims=I64P, pair_row=I64P,
                    pair_out=I64P, item_list=I64P, item_start=I64P, item_count=I64P, o_sc=P32,
                    o_id=I64P, d="i32", sx="i64", sv="i64"),
               dict(BM=bm, BN=bn, BK=bk, IEEE=ieee), 4, 2, dt)
    for p in (1, 8, 64, 256):
        bp = np2(2 * p)
        yield ("_merge_kernel", ift._merge_kernel,
               dict(p_sc=P32, p_id=I64P, scores=P32, out_ids=I64P, n="i32", P="i32"),
               dict(BR=max(1, min(64, 1024 // bp)), BP=bp), 4, 1, "P=%d" % p)
    for nl in (64, 1024, ivf._ROUTE_RESIDENT, ivf._ROUTE_RESIDENT + 1):
        resident = nl <= ivf._ROUTE_RESIDENT
        ct, br = (np2(nl), 2) if resident else (ivf._ROUTE_CHUNK, 1)
        yield ("_select_kernel", ift._select_kernel,
               dict(dot=P32, norms=P32, out_col=I64P, out_key=I64P, R="i32", NCOL="i32", s_d="i64",
                    P="i32"),
               dict(CT=ct, BLOCK_R=br, RESIDENT=resident), 4, 1, "lists=%d" % nl)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arch", choices=ARCHS, action="append")
    args = parser.parse_args()
    worst, bad = {}, []
    for arch in args.arch or ARCHS:
        budget = dp.profile_for_arch(arch).lds_bytes
        target = GPUTarget("hip", arch, 32 if arch.startswith("gfx1") else 64)
        for name, fn, sig, cx, warps, stages, label in cases():
            signature = dict(sig, **{c: "constexpr" for c in cx})
            try:
                kernel = triton.compile(ASTSource(fn=fn, signature=signature, constexprs=cx),
                                        target=target, options=dict(num_warps=warps, num_stages=stages))
                shared = kernel.metadata.shared
            except Exception as error:  # report and continue
                bad.append("%s %s %s: compile error: %s" % (arch, name, label,
                                                           str(error).strip().splitlines()[-1][:160]))
                continue
            key = (arch, name)
            if shared > worst.get(key, (-1, ""))[0]:
                worst[key] = (shared, label)
            if shared > budget:
                bad.append("%s %s %s: %d bytes > %d" % (arch, name, label, shared, budget))
        print("%s (budget %d KiB)" % (arch, budget // 1024))
        for (a, name), (shared, label) in sorted(worst.items()):
            if a == arch:
                print("  %-26s max %6d B  (%s)" % (name, shared, label))
    for line in bad:
        print("OVER/ERROR:", line)
    print("triton", triton.__version__, "-", "all within budget" if not bad else "%d problem(s)" % len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
