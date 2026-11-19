"""W0 smoke matrix: train e-means for each precision x backend and record what happens.

Every case runs in its own subprocess with a timeout, so a hang or crash in one
case (graph capture, an unsupported kernel) cannot take down the others or leave
the GPU busy. Failures are expected on AMD until W1-W5 land: the point is to
record exactly where each combination stops.

    python tools/w0_smoke.py                      # 200k x 64, k=1024 (plan size)
    python tools/w0_smoke.py --rows 20000 --k 256 # smaller, for a shared GPU
    python tools/w0_smoke.py --only bf16          # substring filter on case names

Results go to stdout as a table and to --out (JSON lines).

On ROCm, training uses the eager loop unless EMEANS_ROCM_GRAPH_REPLAY=1 is set
(the variable is inherited by every case). Run the matrix both ways: same
results, and a crash or mismatch with it set means graph replay must stay off.
"""
import argparse
import json
import subprocess
import sys
import time

# name -> (input dtype, quantizer spec, assignment_backend)
CASES = {}
for backend in ("auto", "native", "triton", "cute"):
    CASES["fp32-native/%s" % backend] = ("fp32", "none", backend)
CASES["default-quantizer/auto"] = ("fp32", "default", "auto")
for backend in ("auto", "native", "triton"):
    CASES["bf16-input/%s" % backend] = ("bf16", "none", backend)
    CASES["fp16-input/%s" % backend] = ("fp16", "none", backend)
for backend in ("auto", "triton"):
    CASES["bf16-cast/%s" % backend] = ("fp32", "cast-bf16", backend)
    CASES["fp16-cast/%s" % backend] = ("fp32", "cast-fp16", backend)
    CASES["int8/%s" % backend] = ("fp32", "int8", backend)
    CASES["fp8/%s" % backend] = ("fp32", "fp8", backend)


def run_case(name, rows, d, k, epochs):
    """Executed in the child process. Prints one JSON line."""
    import torch
    from faiss.contrib.e_means.e_means_quantized import QuantizedEmeans
    from faiss.contrib.e_means import quantizer, quantizer_fp8, quantizer_int8

    dtype, qspec, backend = CASES[name]
    out = {"case": name, "hip": torch.version.hip}
    torch.manual_seed(0)
    x = torch.randn(rows, d, device="cuda")
    if dtype == "bf16":
        x = x.to(torch.bfloat16)
    elif dtype == "fp16":
        x = x.to(torch.float16)

    kwargs = dict(nepochs=epochs, assignment_backend=backend)
    if qspec == "none":
        kwargs["quantizer"] = None
    elif qspec == "cast-bf16":
        kwargs["quantizer"] = quantizer.CastQuantizer(d, torch.bfloat16)
    elif qspec == "cast-fp16":
        kwargs["quantizer"] = quantizer.CastQuantizer(d, torch.float16)
    elif qspec == "int8":
        kwargs["quantizer"] = quantizer_int8.Int8Quantizer(d)
    elif qspec == "fp8":
        kwargs["quantizer"] = quantizer_fp8.Fp8Quantizer(d)
    q = kwargs.get("quantizer")
    if q is not None:
        q.train(x)

    model = QuantizedEmeans(d, k, **kwargs)
    torch.cuda.synchronize()
    t0 = time.time()
    centroids, labels = model.train(x)
    torch.cuda.synchronize()
    out["seconds"] = round(time.time() - t0, 3)
    out["backend_used"] = getattr(model, "assignment_backend", None)
    out["centroids_finite"] = bool(torch.isfinite(torch.as_tensor(centroids)).all())
    obj = getattr(model, "obj", None)
    out["obj"] = float(obj.sum()) if obj is not None else None
    out["status"] = "ok"
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=300, help="seconds per case")
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--out", default="w0_smoke_results.jsonl")
    ap.add_argument("--case", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.case:
        run_case(a.case, a.rows, a.d, a.k, a.epochs)
        return

    rows = []
    for name in CASES:
        if a.only not in name:
            continue
        cmd = [sys.executable, __file__, "--case", name, "--rows", str(a.rows), "--d", str(a.d),
               "--k", str(a.k), "--epochs", str(a.epochs)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
            if p.returncode == 0 and lines:
                r = json.loads(lines[-1])
            else:
                err = [l for l in p.stderr.strip().splitlines() if l.strip()]
                r = {"case": name, "status": "FAIL", "rc": p.returncode,
                     "error": (err[-1] if err else "no output")[:240]}
        except subprocess.TimeoutExpired:
            r = {"case": name, "status": "TIMEOUT", "error": "no result in %ds" % a.timeout}
        rows.append(r)
        print("%-26s %-8s %s" % (name, r["status"],
              ("%ss backend=%s obj=%s" % (r.get("seconds"), r.get("backend_used"), r.get("obj")))
              if r["status"] == "ok" else r.get("error", "")), flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
