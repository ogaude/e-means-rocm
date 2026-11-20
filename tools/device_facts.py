"""Record the facts about the current GPU and software stack that e-means relies on.

    python tools/device_facts.py                 # prints JSON, also written to --out
    python tools/device_facts.py --out facts.json

Covers: versions, device properties (arch name, CU / WGP count, LDS, memory), the
Triton target and its shared-memory limit, the e-means device profile, and three
runtime checks done once on gfx1200 for v0.1: torch.addmm(out_dtype=) for FP16/BF16,
CUDA-graph capture with a MemPool, and the BLAS libraries PyTorch can use.
Every check is wrapped, so a failure is recorded instead of stopping the script.
"""
import argparse
import json
import platform
import sys
import traceback

import torch


def _try(fn):
    try:
        return fn()
    except Exception as exc:  # recorded, not raised
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def versions():
    out = {"python": platform.python_version(), "torch": torch.__version__,
           "hip": torch.version.hip, "cuda": torch.version.cuda}
    for name in ("triton", "numpy", "faiss"):
        try:
            out[name] = __import__(name).__version__
        except Exception as exc:
            out[name] = "unavailable (%s)" % type(exc).__name__
    return out


def properties():
    p = torch.cuda.get_device_properties(0)
    keys = ("name", "gcnArchName", "multi_processor_count", "total_memory", "warp_size",
            "shared_memory_per_block", "shared_memory_per_block_optin",
            "shared_memory_per_multiprocessor", "max_threads_per_multi_processor",
            "regs_per_multiprocessor", "L2_cache_size", "major", "minor")
    out = {k: getattr(p, k) for k in keys if hasattr(p, k)}
    out["total_memory_gib"] = round(p.total_memory / 2**30, 1)
    out["device_count"] = torch.cuda.device_count()
    out["get_device_capability"] = list(torch.cuda.get_device_capability(0))
    return out


def triton_target():
    import triton

    driver = triton.runtime.driver.active
    target = driver.get_current_target()
    out = {"backend": target.backend, "arch": str(target.arch),
           "warp_size": getattr(target, "warp_size", None)}
    props = _try(lambda: driver.utils.get_device_properties(driver.get_current_device()))
    if isinstance(props, dict):
        out["device_properties"] = {k: v for k, v in props.items()
                                    if isinstance(v, (int, float, str, bool))}
    return out


def profile():
    from faiss.contrib.e_means.device_profile import device_profile

    p = device_profile(torch.device("cuda", 0))
    return {k: getattr(p, k) for k in p.__dataclass_fields__}


def addmm_out_dtype():
    out = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        def check(dtype=dtype):
            a = torch.randn(256, 96, device="cuda").to(dtype)
            b = torch.randn(96, 128, device="cuda").to(dtype)
            bias = torch.zeros(256, 128, device="cuda")
            r = torch.addmm(bias, a, b, out_dtype=torch.float32)
            ref = a.float() @ b.float()
            return {"ok": r.dtype == torch.float32,
                    "max_abs_err": float((r - ref).abs().max())}
        out[name] = _try(check)
    return out


def graph_capture():
    x = torch.randn(1024, 64, device="cuda")
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = x @ x.T  # warm-up outside capture
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
        y = x @ x.T
    x.add_(1.0)
    g.replay()
    torch.cuda.synchronize()
    return {"ok": bool(torch.allclose(y, x @ x.T, rtol=1e-4, atol=1e-3))}


def blas():
    out = {"preferred": str(_try(torch.backends.cuda.preferred_blas_library))}
    for lib in ("cublas", "cublaslt"):  # rocBLAS / hipBLASLt on ROCm
        def check(lib=lib):
            torch.backends.cuda.preferred_blas_library(lib)
            a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
            (a @ a).sum().item()
            return "ok"
        out[lib] = _try(check)
    _try(lambda: torch.backends.cuda.preferred_blas_library("default"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="device_facts.json")
    a = ap.parse_args()
    facts = {"versions": _try(versions), "cuda_available": torch.cuda.is_available()}
    if facts["cuda_available"]:
        for name, fn in (("properties", properties), ("triton", triton_target),
                         ("profile", profile), ("addmm_out_dtype", addmm_out_dtype),
                         ("graph_capture", graph_capture), ("blas", blas)):
            try:
                facts[name] = fn()
            except Exception as exc:
                facts[name] = {"error": "%s: %s" % (type(exc).__name__, exc),
                               "trace": traceback.format_exc(limit=3)}
    text = json.dumps(facts, indent=2, default=str)
    print(text)
    with open(a.out, "w") as f:
        f.write(text + "\n")
    return 0 if facts["cuda_available"] else 1


if __name__ == "__main__":
    sys.exit(main())
