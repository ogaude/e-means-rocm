"""Ask before training: which backend would run, and how much GPU memory it needs.

* ``supported(device, dtype, ...)`` returns the assignment backend that training
  would use for a representation on a device ("native", "triton" or "cute"), or
  ``None`` when training would be refused. It checks the same device facts and
  installed software as training, without running a kernel.
* ``memory_estimate(model, n, dtype, device)`` returns an upper bound, in bytes, on
  the GPU memory an untrained ``Emeans`` or ``QuantizedEmeans(quantizer=None)``
  needs to ``train`` on ``n`` rows, so a caller can choose between ``train`` and the
  streaming loop (``begin``/``accumulate``/``update``/``finalize``) up front.
"""

from types import SimpleNamespace

import torch

from .device_profile import (
    amd_native_reduced_supported,
    amd_triton_supported,
    is_rocm,
    nvidia_capability,
)

__all__ = ["supported", "memory_estimate"]

_FLOAT8 = getattr(torch, "float8_e4m3fn", None)
_FLOATING = (torch.float32, torch.float16, torch.bfloat16)
_BYTES = (torch.int8, torch.uint8)
_BACKENDS = ("auto", "native", "triton", "cute")


def _representations():
    return _FLOATING + _BYTES + ((_FLOAT8,) if _FLOAT8 is not None else ())


def _accepts(name, device, dtype, spherical, d):
    """Whether an explicit backend runs this representation (device facts only)."""
    if device.type != "cuda":
        return name == "native"
    capability = nvidia_capability(device)
    if name == "native":
        if dtype is torch.float32:
            return True
        if dtype not in _FLOATING:
            return False
        if is_rocm():
            return amd_native_reduced_supported(device)
        return capability >= (8, 0) and "out_dtype" in (torch.addmm.__doc__ or "")
    if spherical and dtype not in _FLOATING:
        return False
    if name == "triton":
        if is_rocm():
            if not amd_triton_supported(device):
                return False
        elif capability < (8, 0):
            return False
        try:
            if dtype in _BYTES:
                from .quantizer_int8 import _check_int8_support

                _check_int8_support(device)
                return d is None or d <= 43919
            if dtype is _FLOAT8:
                from .quantizer_fp8 import _check_fp8_support

                _check_fp8_support(device)
        except (RuntimeError, ValueError):
            return False
        return True
    # CuTe: NVIDIA SM90 only; BF16, E4M3 and bytes; spherical only plain BF16.
    if is_rocm() or capability != (9, 0):
        return False
    if spherical:
        return dtype is torch.bfloat16
    return dtype is torch.bfloat16 or dtype is _FLOAT8 or dtype in _BYTES


def supported(device, dtype, *, spherical=False, backend="auto", d=None, k=None):
    """Backend name training would use for ``dtype`` rows on ``device``, or None.

    ``dtype`` is the representation the assignment sees: the dtype of the rows you
    pass (FP32, FP16, BF16, ``float8_e4m3fn``, INT8, UINT8), or the quantizer's
    dtype. Raw INT8/UINT8 rows are trained with ``QuantizedEmeans``. FP32 rows given
    to ``Emeans`` on a GPU are converted to BF16 by default: ask for
    ``torch.bfloat16`` there, or for ``torch.float32`` with
    ``QuantizedEmeans(quantizer=None)``. On the CPU every representation is
    converted to FP32 and runs on "native".

    ``backend`` is the requested ``assignment_backend``. ``d`` and ``k``, when
    given, also apply the extent limits (INT8 needs d <= 43,919). Missing or broken
    software for the selected backend (Triton, the CuTe stack) gives None, as
    training would raise. The answer comes from device facts and installed
    packages; it does not compile or run a kernel, so a broken GPU driver can still
    fail at training time.
    """
    if backend not in _BACKENDS:
        raise ValueError("backend must be auto, native, triton or cute")
    device = torch.device(device)
    if dtype not in _representations():
        return None
    if device.type == "cuda" and torch.cuda.device_count() and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if backend == "auto":
        from .torch_backend import select_backend

        try:
            name = select_backend(SimpleNamespace(device=device), dtype,
                                  k or 1, d or 1, "auto").name
        except (RuntimeError, ValueError, ImportError):
            return None
    else:
        name = backend
    if not _accepts(name, device, dtype, spherical, d):
        return None
    try:
        from .torch_backend import CuteBackend, TorchBackend, TritonBackend

        cls = {"native": TorchBackend, "triton": TritonBackend, "cute": CuteBackend}[name]
        cls(4096).require_training(device)
    except (RuntimeError, ImportError, ValueError):
        return None
    return name


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
    are not covered. Returns 0 for a CPU device. Raises like training if the
    representation is not supported there.
    """
    from .base import _resolve_batch_law, _resolve_recipe

    device = torch.device(device)
    if device.type != "cuda":
        return 0
    n, d, k = int(n), int(model.d), int(model.k)
    if n < 1:
        raise ValueError("n must be >= 1")
    if getattr(model, "quantizer", None) is not None:
        raise ValueError("memory_estimate covers models without a quantizer")
    compute = dtype
    automatic = getattr(model, "_automatic", True)
    if dtype is torch.float32 and automatic:
        from .e_means_torch import _default_cast_supported

        if _default_cast_supported(device, model.assignment_backend):
            compute = torch.bfloat16
    backend = supported(device, compute, spherical=model.spherical,
                        backend=model.assignment_backend, d=d, k=k)
    if backend is None:
        raise RuntimeError("training does not support %s rows with assignment_backend=%r "
                           "on this device" % (compute, model.assignment_backend))
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
