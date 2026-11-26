"""Vendor-aware GPU conditions for e-means test skips.

ROCm PyTorch reports AMD GPUs as device type "cuda", and on ROCm
torch.cuda.get_device_capability() returns gfx numbers ((9, 4) for gfx942,
(12, 0) for gfx1200), so "capability >= (8, 0)" alone would wrongly match AMD.
Skips use these instead:

* nvidia_sm80()      NVIDIA SM80 or later (the upstream condition, exactly).
* amd_triton_gpu()   AMD target family (gfx942, gfx950, gfx11xx, gfx120x) with Triton.
* triton_gpu()       either of the above with Triton: the Triton backend runs here.
* amd_fp8_gpu()      AMD family with a native E4M3 matrix instruction (gfx950, gfx120x).
* native_reduced_gpu() native PyTorch FP16/BF16 assignment runs here.
* nvidia_sm(*caps)   an NVIDIA GPU whose SM is one of caps.
* int8_gpu()         INT8 assignment runs here: NVIDIA SM80/SM90/SM100 (upstream) or an
                     AMD target family.
* fp8_gpu()          E4M3 assignment runs here: NVIDIA SM90/SM100/SM103 (upstream) or
                     gfx950/gfx120x.
* cute_gpu()         the CuTe backend's NVIDIA SM90 (never AMD).
* graph_replay_gpu() a GPU on which training uses CUDA-graph replay (on ROCm only
                     with EMEANS_ROCM_GRAPH_REPLAY=1; see device_profile).

Each is a function evaluated when called; module-level flags call them once.
"""
import importlib.util

import torch


def _triton_installed():
    return importlib.util.find_spec("triton") is not None


def nvidia_sm80():
    return (torch.cuda.is_available() and torch.version.hip is None
            and torch.cuda.get_device_capability() >= (8, 0))


def amd_triton_gpu():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return False
    from faiss.contrib.e_means.device_profile import amd_triton_supported

    return amd_triton_supported("cuda")


def amd_fp8_gpu():
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return False
    from faiss.contrib.e_means.device_profile import amd_fp8_supported

    return amd_fp8_supported("cuda")


def nvidia_sm(*capabilities):
    return (torch.cuda.is_available() and torch.version.hip is None
            and torch.cuda.get_device_capability() in capabilities)


def int8_gpu():
    return nvidia_sm((8, 0), (9, 0), (10, 0)) or amd_triton_gpu()


def fp8_gpu():
    return nvidia_sm((9, 0), (10, 0), (10, 3)) or amd_fp8_gpu()


def cute_gpu():
    return nvidia_sm((9, 0))


def triton_gpu():
    return (nvidia_sm80() and _triton_installed()) or amd_triton_gpu()


def native_reduced_gpu():
    """Native PyTorch FP16/BF16 assignment runs here (addmm with out_dtype=float32):
    NVIDIA SM80+ (upstream), or an AMD target family."""
    if nvidia_sm80():
        return True
    if not (torch.cuda.is_available() and torch.version.hip is not None):
        return False
    from faiss.contrib.e_means.device_profile import amd_native_reduced_supported

    return amd_native_reduced_supported("cuda")


def graph_replay_gpu():
    if not torch.cuda.is_available():
        return False
    from faiss.contrib.e_means.device_profile import graph_replay_supported

    return graph_replay_supported("cuda")
