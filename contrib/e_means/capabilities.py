"""Ask before training: which backend would run, and how much GPU memory it needs.

* ``supported(device, dtype, ...)`` returns the assignment backend that training
  would use for a representation on a device ("native", "triton" or "cute"), or
  ``None`` when training would be refused. It checks the same device facts and
  installed software as training, without running a kernel. ``explain=True`` also
  gives the reason for a refusal.
* ``memory_estimate(model, n, dtype, device)`` returns an upper bound, in bytes, on
  the GPU memory an untrained ``Emeans`` or ``QuantizedEmeans(quantizer=None)``
  needs to ``train`` on ``n`` rows, so a caller can choose between ``train`` and the
  streaming loop (``begin``/``accumulate``/``update``/``finalize``) up front.

Both answer without PyTorch installed: ``supported`` refuses with the reason "torch
is not installed", and ``memory_estimate`` returns 0 for the CPU and raises for a
GPU. Devices and dtypes may be given as strings ("cuda:0", "bfloat16") for that case.
"""

from types import SimpleNamespace

try:
    import torch
except ImportError:  # CPU-only install: NumpyEmeans works, the torch path does not.
    torch = None

__all__ = ["supported", "memory_estimate"]

_BACKENDS = ("auto", "native", "triton", "cute")
_NAMES = ("float32", "float16", "bfloat16", "int8", "uint8", "float8_e4m3fn")
_NO_TORCH = "torch is not installed"


def _torch_dtypes():
    floating = (torch.float32, torch.float16, torch.bfloat16)
    fp8 = getattr(torch, "float8_e4m3fn", None)
    return floating, (torch.int8, torch.uint8), fp8


def _as_dtype(dtype):
    """A torch dtype from a torch dtype or its name ("bfloat16"); None if unknown."""
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype, None) if dtype in _NAMES else None
    floating, nbytes, fp8 = _torch_dtypes()
    if dtype in floating or dtype in nbytes or (fp8 is not None and dtype is fp8):
        return dtype
    return None


def _device_type(device):
    if torch is not None:
        return torch.device(device).type
    if isinstance(device, int):  # as torch.device(0): GPU 0
        return "cuda"
    return str(device).split(":", 1)[0]


def _refusal(name, device, dtype, spherical, d):
    """Why an explicit backend cannot run this representation, or None if it can."""
    from .device_profile import (
        AMD_TARGETS,
        amd_native_reduced_supported,
        amd_triton_supported,
        is_rocm,
        nvidia_capability,
    )

    floating, nbytes, fp8 = _torch_dtypes()
    if device.type != "cuda":
        return None if name == "native" else "%s runs only on a GPU" % name
    capability = nvidia_capability(device)
    if name == "native":
        if dtype is torch.float32:
            return None
        if dtype not in floating:
            return "native takes FP32, FP16 or BF16 rows on a GPU"
        if is_rocm():
            if amd_native_reduced_supported(device):
                return None
            return "native FP16/BF16 on AMD needs a %s GPU and torch.addmm(out_dtype=)" % AMD_TARGETS
        if capability >= (8, 0) and "out_dtype" in (torch.addmm.__doc__ or ""):
            return None
        return "native FP16/BF16 needs an NVIDIA SM80 or later GPU and torch.addmm(out_dtype=)"
    if spherical and dtype not in floating:
        return "spherical training takes FP32, FP16 or BF16 rows"
    if name == "triton":
        if is_rocm():
            if not amd_triton_supported(device):
                return "Triton assignment on AMD needs a %s GPU and Triton installed" % AMD_TARGETS
        elif capability < (8, 0):
            return "Triton assignment needs an NVIDIA SM80 or later GPU"
        try:
            if dtype in nbytes:
                from .quantizer_int8 import _check_int8_support

                _check_int8_support(device)
                if d is not None and d > 43919:
                    return "INT8 assignment needs d <= 43,919"
            elif dtype is fp8:
                from .quantizer_fp8 import _check_fp8_support

                _check_fp8_support(device)
        except (RuntimeError, ValueError) as error:
            return str(error)
        return None
    # CuTe: NVIDIA SM90 only; BF16, E4M3 and bytes; spherical only plain BF16.
    if is_rocm() or capability != (9, 0):
        return "CuTe assignment needs an NVIDIA SM90 GPU"
    if spherical:
        return None if dtype is torch.bfloat16 else "spherical CuTe assignment takes BF16 rows"
    if dtype is torch.bfloat16 or dtype is fp8 or dtype in nbytes:
        return None
    return "CuTe assignment takes BF16, FP8 E4M3 or INT8/UINT8 rows"


def _supported(device, dtype, spherical, backend, d, k):
    """(backend name, None) when training would run, else (None, reason)."""
    if torch is None:
        return None, _NO_TORCH
    representation = _as_dtype(dtype)
    if representation is None:
        return None, ("%r is not a representation e-means trains (FP32, FP16, BF16, "
                      "float8_e4m3fn, INT8, UINT8)" % (dtype,))
    device = torch.device(device)
    try:
        if device.type == "cuda" and torch.cuda.device_count() and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if backend == "auto":
            from .torch_backend import select_backend

            try:
                name = select_backend(SimpleNamespace(device=device), representation,
                                      k or 1, d or 1, "auto").name
            except (RuntimeError, ValueError, ImportError, AssertionError) as error:
                return None, str(error)
        else:
            name = backend
        reason = _refusal(name, device, representation, spherical, d)
    except (RuntimeError, AssertionError) as error:
        # No usable GPU: torch raises RuntimeError("No CUDA GPUs are available") on a GPU
        # build with no visible GPU, AssertionError("Torch not compiled with CUDA
        # enabled") on a CPU-only build.
        return None, str(error)
    if reason is not None:
        return None, reason
    try:
        from .torch_backend import CuteBackend, TorchBackend, TritonBackend

        cls = {"native": TorchBackend, "triton": TritonBackend, "cute": CuteBackend}[name]
        cls(4096).require_training(device)
    except (RuntimeError, ImportError, ValueError) as error:
        return None, str(error)
    return name, None


def supported(device, dtype, *, spherical=False, backend="auto", d=None, k=None,
              explain=False):
    """Backend name training would use for ``dtype`` rows on ``device``, or None.

    ``dtype`` is the representation the assignment sees: the dtype of the rows you
    pass (FP32, FP16, BF16, ``float8_e4m3fn``, INT8, UINT8), or the quantizer's
    dtype, as a torch dtype or its name (``"bfloat16"``). Raw INT8/UINT8 rows are
    trained with ``QuantizedEmeans``. FP32 rows given to ``Emeans`` on a GPU are
    converted to BF16 by default: ask for ``torch.bfloat16`` there, or for
    ``torch.float32`` with ``QuantizedEmeans(quantizer=None)``. On the CPU every
    representation is converted to FP32 and runs on "native". ``device`` is a torch
    device or a string ("cpu", "cuda", "cuda:1").

    ``backend`` is the requested ``assignment_backend``. ``d`` and ``k``, when
    given, also apply the extent limits (INT8 needs d <= 43,919). Missing or broken
    software for the selected backend (Triton, the CuTe stack) gives None, as
    training would raise. The answer comes from device facts and installed
    packages; it does not compile or run a kernel, so a broken GPU driver can still
    fail at training time.

    Without PyTorch installed, ``Emeans`` and ``QuantizedEmeans`` cannot train on
    any device (``NumpyEmeans`` is the CPU path without torch), so the answer is
    None. With ``explain=True`` the result is a pair ``(backend, reason)``:
    ``("triton", None)`` when training would run, or ``(None, "torch is not
    installed")`` and similar when it would be refused.
    """
    if backend not in _BACKENDS:
        raise ValueError("backend must be auto, native, triton or cute")
    name, reason = _supported(device, dtype, spherical, backend, d, k)
    return (name, reason) if explain else name


# Measured on gfx1200 with ROCm 10.1: the first torch.dot allocates a 32 MiB BLAS
# workspace and the first FP32/FP16/BF16 addmm (native assignment) another 80 MiB.
_BLAS_WORKSPACE = 32 * 2**20
_ADDMM_WORKSPACE = 96 * 2**20
_MARGIN = 1.25


def _element_size(dtype):
    return torch.empty(0, dtype=dtype).element_size()


def memory_estimate(model, n, dtype, device, *, include_input=True):
    """Upper bound in bytes on the GPU memory ``model.train(x)`` needs.

    ``model`` is an untrained ``Emeans`` or ``QuantizedEmeans(quantizer=None)``;
    its own settings (``k``, ``d``, ``nepochs``, ``batch_size``, ``batch_law``,
    chunk sizes, ``capture_assignments``, ``assignment_backend``) are used. ``x``
    would have ``n`` rows of ``dtype`` on ``device``. With ``include_input=False``
    the n x d input itself is left out, for data already on the GPU.

    The bound adds the input, per-row buffers (shuffle order, captured labels),
    centroid state, one statistical batch's gathered and converted rows and
    statistics workspace, the native distance tile, and BLAS workspaces measured
    on ROCm, then a 25% margin on everything but the input. Training with a quantizer (FP8, INT8, centered
    casts), ``max_points_per_centroid`` (which copies the subset) and several GPUs
    are not covered. Returns 0 for a CPU device, also without PyTorch installed.
    Raises a RuntimeError like training if the representation is not supported
    there, including "not supported: torch is not installed" for a GPU device.
    ``dtype`` and ``device`` may be given as strings, as for ``supported``.
    """
    if _device_type(device) != "cuda":
        return 0
    if torch is None:
        raise RuntimeError("not supported: %s; GPU training needs PyTorch" % _NO_TORCH)
    from .base import _resolve_batch_law, _resolve_recipe

    device = torch.device(device)
    named = _as_dtype(dtype)
    dtype = named if named is not None else dtype
    n, d, k = int(n), int(model.d), int(model.k)
    if n < 1:
        raise ValueError("n must be >= 1")
    if getattr(model, "quantizer", None) is not None:
        raise ValueError("memory_estimate covers models without a quantizer")
    if not torch.cuda.is_available():
        raise RuntimeError("not supported: torch sees no GPU (a CPU-only build, or none visible)")
    compute = dtype
    automatic = getattr(model, "_automatic", True)
    if dtype is torch.float32 and automatic:
        from .e_means_torch import _default_cast_supported

        if _default_cast_supported(device, model.assignment_backend):
            compute = torch.bfloat16
    backend, reason = supported(device, compute, spherical=model.spherical,
                                backend=model.assignment_backend, d=d, k=k, explain=True)
    if backend is None:
        raise RuntimeError("training does not support %s rows with assignment_backend=%r "
                           "on this device: %s" % (compute, model.assignment_backend, reason))
    law, _ = _resolve_batch_law(model.batch_law, d)
    batch, _ = _resolve_recipe(n, k, model.nepochs, law, model.batch_size, model.alpha0)
    if model.assignment_chunk_size is not None:
        chunk = model.assignment_chunk_size
    else:
        width = max(d, min(k, model.centroid_chunk_size)) if backend == "native" else d
        chunk = min(2**20, max(8192, 2**26 // width))
    rows = min(batch, chunk)
    stored, computed = _element_size(dtype), _element_size(compute)

    total = 4 * 8 * n                                # shuffle order and randperm scratch
    if model.capture_assignments:
        total += 8 * n
    total += 8 * k * (d + 1) * 4                      # centroids, stats, sums, readouts
    total += rows * (d + 1) * 4                       # statistics workspace
    total += rows * d * stored                        # gathered batch rows
    if compute is not dtype:
        total += rows * d * computed                  # converted batch rows
    total += rows * 32                                # distances, labels, norms
    total += _BLAS_WORKSPACE
    if backend == "native":
        total += rows * min(k, model.centroid_chunk_size) * 4
        total += _ADDMM_WORKSPACE
    return int(total * _MARGIN) + (n * d * stored if include_input else 0)
