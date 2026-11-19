"""Vendor and architecture-family facts for the GPU an e-means tensor lives on.

ROCm PyTorch reports AMD GPUs as device type "cuda", so ``tensor.is_cuda`` does not
tell NVIDIA from AMD. This module is the one place that does:

* ``nvidia_capability(device)`` is ``torch.cuda.get_device_capability(device)`` on
  non-ROCm builds and ``None`` on ROCm builds. Use it instead of the torch call for
  NVIDIA-only feature gates (TMA descriptors, CuTe, FP8 tensor cores). It is not
  cached, so tests that mock ``torch.cuda.get_device_capability`` keep working.
* ``device_profile(device)`` returns a ``DeviceProfile`` with the vendor and, on AMD,
  the architecture family and the facts kernel selection and tuning need.
* ``profile_for_arch(arch)`` is the pure AMD lookup behind it, testable without a GPU.

AMD families are matched by architecture prefix, not exact name, so siblings such as
gfx1101 or gfx1201 get their family's profile. An AMD GPU outside the four target
families gets ``family=None``; callers treat that as "no Triton fast path".
"""

import functools
from dataclasses import dataclass

import torch

KIB = 1024

# Family facts. ``lds_bytes`` is the shared memory (LDS) a single workgroup may use.
# Confirmed: gfx942 64 KiB (upstream comment in triton_assignment.py, ROCm Triton) and
# gfx1200 64 KiB (shared_memory_per_block and Triton max_shared_mem on an RX 9060 XT,
# ROCm 10.1). cdna4 and rdna3 use a conservative 64 KiB until measured on hardware,
# which keeps per-family tile tables (W6) inside a budget that is known to exist.
#
# ``fp32_matrix``: an FP32 matrix instruction exists (MFMA on CDNA; RDNA has none,
# so FP32 tl.dot is scalar there).
# ``native_fp8``: an OCP E4M3 (torch.float8_e4m3fn / tl.float8e4nv) matrix
# instruction exists. gfx942 has FP8 MFMA only for the FNUZ variant, so Triton
# emulates E4M3FN through FP16 there.
_FAMILIES = {
    "cdna3": dict(wavefront=64, lds_bytes=64 * KIB, lds_confirmed=True,
                  fp32_matrix=True, native_fp8=False),
    "cdna4": dict(wavefront=64, lds_bytes=64 * KIB, lds_confirmed=False,
                  fp32_matrix=True, native_fp8=True),
    "rdna3": dict(wavefront=32, lds_bytes=64 * KIB, lds_confirmed=False,
                  fp32_matrix=False, native_fp8=False),
    "rdna4": dict(wavefront=32, lds_bytes=64 * KIB, lds_confirmed=True,
                  fp32_matrix=False, native_fp8=True),
}

# Longest prefix wins. gfx94x = MI300 series, gfx950 = MI350 series,
# gfx11xx = RDNA3/3.5 (RX 7000, APUs), gfx120x = RDNA4 (RX 9000).
_PREFIXES = (
    ("gfx94", "cdna3"),
    ("gfx950", "cdna4"),
    ("gfx11", "rdna3"),
    ("gfx120", "rdna4"),
)


@dataclass(frozen=True)
class DeviceProfile:
    vendor: str                      # "nvidia", "amd" or "cpu"
    arch: str = None                 # "gfx1200", "sm90", ...
    capability: tuple = None         # NVIDIA SM capability, None elsewhere
    family: str = None               # AMD: "cdna3", "cdna4", "rdna3", "rdna4" or None
    wavefront: int = None
    lds_bytes: int = None
    lds_confirmed: bool = False
    fp32_matrix: bool = False
    native_fp8: bool = False

    @property
    def is_nvidia(self):
        return self.vendor == "nvidia"

    @property
    def is_amd(self):
        return self.vendor == "amd"

    @property
    def known_amd(self):
        """AMD GPU in one of the four families e-means has kernels tuned for."""
        return self.is_amd and self.family is not None


def normalize_arch(arch):
    """'gfx90a:sramecc+:xnack-' -> 'gfx90a'; lower-case; None stays None."""
    if not arch:
        return None
    return str(arch).split(":", 1)[0].strip().lower() or None


def family_of(arch):
    """AMD architecture family for an arch name, or None when not a target family."""
    arch = normalize_arch(arch)
    if arch is None:
        return None
    best = None
    for prefix, family in _PREFIXES:
        if arch.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, family)
    return best[1] if best else None


def profile_for_arch(arch):
    """AMD profile for an arch name. Pure: needs no GPU."""
    arch = normalize_arch(arch)
    family = family_of(arch)
    if family is None:
        return DeviceProfile(vendor="amd", arch=arch)
    return DeviceProfile(vendor="amd", arch=arch, family=family, **_FAMILIES[family])


def is_rocm():
    return torch.version.hip is not None


def _device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def nvidia_capability(device):
    """SM capability on NVIDIA (non-ROCm) builds; None on ROCm builds.

    On non-ROCm builds this is exactly ``torch.cuda.get_device_capability(device)``:
    uncached and with no device-type check, as the call sites did before this helper
    existed, so NVIDIA behaviour and tests that mock the capability are unchanged.
    """
    if is_rocm():
        return None
    return torch.cuda.get_device_capability(device)


@functools.lru_cache(maxsize=None)
def _amd_arch(index):
    properties = torch.cuda.get_device_properties(index)
    arch = getattr(properties, "gcnArchName", None)
    if not arch:
        # Fall back to the Triton target of the current device.
        try:
            import triton

            arch = triton.runtime.driver.active.get_current_target().arch
        except Exception:
            arch = None
    return normalize_arch(arch)


@functools.lru_cache(maxsize=None)
def triton_available():
    """Whether the triton package is importable (checked once, without importing it)."""
    import importlib.util

    return importlib.util.find_spec("triton") is not None


def amd_triton_supported(device):
    """True on ROCm for a GPU in a target family, with Triton installed.

    This is the condition under which automatic selection uses the Triton backend
    and the default FP32 -> BF16 cast applies on AMD.
    """
    if not is_rocm() or torch.device(device).type != "cuda":
        return False
    return device_profile(device).known_amd and triton_available()


def addmm_out_dtype_available():
    """Whether this PyTorch's addmm accepts out_dtype (native FP16/BF16 assignment)."""
    return "out_dtype" in (torch.addmm.__doc__ or "")


def amd_native_reduced_supported(device):
    """True on ROCm for a target-family GPU whose PyTorch has addmm(out_dtype=).

    Native FP16/BF16 assignment (torch.addmm with out_dtype=float32) is verified on
    gfx1200 with ROCm 10.1. As on NVIDIA, where it needs SM80+, it is limited to the
    target families; other AMD GPUs stay on native FP32.
    """
    if not is_rocm() or torch.device(device).type != "cuda":
        return False
    return device_profile(device).known_amd and addmm_out_dtype_available()


AMD_TARGETS = "gfx942, gfx950, gfx11xx or gfx120x"

GRAPH_REPLAY_ENV = "EMEANS_ROCM_GRAPH_REPLAY"


def graph_replay_supported(device):
    """Whether training may use CUDA-graph capture and replay on this device.

    NVIDIA: always, as upstream. ROCm: off by default; set EMEANS_ROCM_GRAPH_REPLAY=1
    to opt in. With replay off, training runs the existing eager loop, which gives
    identical results. Verified on gfx1200 (RX 9060 XT, ROCm 10.1): capture and replay
    work, all replay tests pass, and deterministic-mode runs are bit-identical to eager,
    but replay gave no measurable speed-up there, so eager stays the default until a
    family shows a gain. ``device`` is accepted for that per-family decision.
    """
    if not is_rocm():
        return True
    import os

    return os.environ.get(GRAPH_REPLAY_ENV, "0") == "1"


def device_profile(device):
    """Profile of the device a tensor lives on (``tensor.device`` or a device spec)."""
    device = _device(device)
    if device.type != "cuda":
        return DeviceProfile(vendor="cpu" if device.type == "cpu" else device.type)
    if is_rocm():
        return profile_for_arch(_amd_arch(device.index))
    capability = tuple(torch.cuda.get_device_capability(device))
    return DeviceProfile(vendor="nvidia", arch="sm%d%d" % capability, capability=capability)
