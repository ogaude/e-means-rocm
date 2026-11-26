"""Warm train() benchmark: Triton vs PyTorch backend, eager vs graph replay.

    python tools/bench_train.py                                   # default shapes
    python tools/bench_train.py --shapes 200000x64x1024 --precisions bf16
    python tools/bench_train.py --replay 0 1                      # eager and replay

Each case (shape x precision x backend x replay) runs in its own subprocess with a
timeout. Inside it, one short train() compiles the kernels, then train() with
--epochs epochs is timed --repeat times and the fastest run is kept ("warm").

Precisions: bf16 = FP32 input with the default FP32 -> BF16 cast (what
Emeans(d, k).train(x) does); fp16 = FP32 input cast to FP16; fp32 = FP32 data with
no quantizer; int8 / fp8 = FP32 data encoded once with a calibrated Int8Quantizer /
Fp8Quantizer, trained on the codes (Triton only: the PyTorch backend has no byte
path; encoding is done before timing, while bf16/fp16 convert during training).
Backends: triton, native (PyTorch). Replay 1 sets EMEANS_ROCM_GRAPH_REPLAY=1 (ignored
on NVIDIA, where replay is always on). --no-tables clears the per-family AMD tuning
tables (direct, INT8, split-K), so the NVIDIA-fitted settings run: compare a run with
and without it to measure what the tuning gains.

Results: a table on stdout and JSON lines in --out.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def run_case(rows, d, k, precision, backend, epochs, repeat, no_tables=False):
    import torch
    if no_tables:
        from faiss.contrib.e_means import triton_assignment as ta
        for table in (ta._AMD_DIRECT_CONFIGS, ta._AMD_INT8_CONFIGS,
                      ta._AMD_INT8_SMALL_CONFIGS, ta._AMD_SPLIT_K_MAX_ROW_BLOCKS,
                      ta._AMD_SPLIT_K_STAGES, ta._AMD_DIRECT_SECOND_CONFIGS,
                      ta._AMD_INT8_SECOND_CONFIGS):
            table.clear()
    from faiss.contrib.e_means import quantizer
    from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
    from faiss.contrib.e_means.quantizer_fp8 import Fp8Quantizer
    from faiss.contrib.e_means.quantizer_int8 import Int8Quantizer

    torch.manual_seed(0)
    x = torch.randn(rows, d, device="cuda")
    data, byte_q = x, None
    if precision in ("int8", "fp8"):
        byte_q = (Int8Quantizer if precision == "int8" else Fp8Quantizer)(d)
        byte_q.train(x)
        data = byte_q.compute_codes(x)  # byte quantizers train on their codes

    def train(nepochs):
        kwargs = dict(assignment_backend=backend)
        if byte_q is not None:
            kwargs["quantizer"] = byte_q
        elif precision == "fp32":
            kwargs["quantizer"] = None
        elif precision == "fp16":
            kwargs["quantizer"] = quantizer.CastQuantizer(d, torch.float16)
            kwargs["quantizer"].train(x)
        # bf16: default quantizer (FP32 -> BF16 cast), as Emeans(d, k).train(x)
        model = QuantizedEmeans(d, k, nepochs=nepochs, **kwargs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train(data)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, model

    train(1)  # compile and warm up
    times, model = [], None
    for _ in range(repeat):
        t, model = train(epochs)
        times.append(t)
    recipe = getattr(model, "recipe", None) or {}
    obj = getattr(model, "obj", None)
    print(json.dumps({
        "seconds": round(min(times), 4), "all_seconds": [round(t, 4) for t in times],
        "backend_used": recipe.get("assignment_backend"),
        "representation": recipe.get("representation"),
        "obj": float(obj.sum()) if obj is not None else None,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }))


def parse_shape(text):
    rows, d, k = (int(v) for v in text.lower().split("x"))
    return rows, d, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+",
                    default=["200000x64x1024", "1000000x128x4096", "1000000x768x4096",
                             "4000000x128x16384"],
                    help="ROWSxDxK")
    ap.add_argument("--precisions", nargs="+", default=["bf16", "fp16", "fp32"])
    ap.add_argument("--backends", nargs="+", default=["triton", "native"])
    ap.add_argument("--replay", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per case")
    ap.add_argument("--out", default="bench_train.jsonl")
    ap.add_argument("--no-tables", action="store_true",
                    help="clear the AMD tuning tables (NVIDIA-fitted settings)")
    ap.add_argument("--case", help=argparse.SUPPRESS)
    a = ap.parse_args()
    unknown = set(a.precisions) - {"bf16", "fp16", "fp32", "int8", "fp8"}
    if unknown:
        ap.error("unknown precision(s): %s" % ", ".join(sorted(unknown)))

    if a.case:
        shape, precision, backend = a.case.split(",")
        run_case(*parse_shape(shape), precision, backend, a.epochs, a.repeat, a.no_tables)
        return

    results = []
    with open(a.out, "w") as f:
        for shape in a.shapes:
            for precision in a.precisions:
                for replay in a.replay:
                    for backend in a.backends:
                        if precision in ("int8", "fp8") and backend != "triton":
                            continue  # no PyTorch byte path
                        env = dict(os.environ, EMEANS_ROCM_GRAPH_REPLAY=str(replay))
                        cmd = [sys.executable, __file__, "--case",
                               "%s,%s,%s" % (shape, precision, backend),
                               "--epochs", str(a.epochs), "--repeat", str(a.repeat)]
                        if a.no_tables:
                            cmd.append("--no-tables")
                        r = {"shape": shape, "precision": precision, "backend": backend,
                             "replay": replay, "epochs": a.epochs, "tables": not a.no_tables}
                        try:
                            p = subprocess.run(cmd, capture_output=True, text=True,
                                               timeout=a.timeout, env=env)
                            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
                            if p.returncode == 0 and lines:
                                r.update(json.loads(lines[-1]), status="ok")
                            else:
                                err = [l for l in p.stderr.splitlines() if l.strip()]
                                r.update(status="FAIL", rc=p.returncode,
                                         error=(err[-1] if err else "no output")[:240])
                        except subprocess.TimeoutExpired:
                            r.update(status="TIMEOUT")
                        results.append(r)
                        f.write(json.dumps(r) + "\n")
                        f.flush()
                        print("%-20s %-4s replay=%d %-6s %s" % (
                            shape, precision, replay, backend,
                            "%.3f s  backend=%s peak=%s GiB" % (
                                r["seconds"], r["backend_used"], r["peak_gib"])
                            if r["status"] == "ok" else "%s %s" % (r["status"], r.get("error", ""))),
                            flush=True)

    # Summary: Triton speed-up over native, and replay speed-up over eager.
    def t(shape, precision, backend, replay):
        for r in results:
            if (r["shape"], r["precision"], r["backend"], r["replay"]) == (
                    shape, precision, backend, replay) and r["status"] == "ok":
                return r["seconds"]
        return None

    print("\n%-20s %-4s %12s %12s %12s %12s %9s %9s" % (
        "shape", "prec", "triton eager", "native eager", "triton rep", "native rep",
        "tri/nat", "rep gain"))
    for shape in a.shapes:
        for precision in a.precisions:
            te, ne = t(shape, precision, "triton", 0), t(shape, precision, "native", 0)
            tr, nr = t(shape, precision, "triton", 1), t(shape, precision, "native", 1)
            fmt = lambda v: "%12.3f" % v if v else "%12s" % "-"
            ratio = lambda num, den: "%8.2fx" % (num / den) if num and den else "%9s" % "-"
            print("%-20s %-4s %s %s %s %s %s %s" % (
                shape, precision, fmt(te), fmt(ne), fmt(tr), fmt(nr),
                ratio(ne, te), ratio(te, tr)))


if __name__ == "__main__":
    main()
