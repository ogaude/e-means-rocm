# e-means-rocm

AMD GPU (ROCm) support for [e-means](contrib/e_means/README.md), the mini-batch k-means in
faiss `contrib/e_means`. The goal is that `Emeans(d, k).train(x)` runs on AMD GPUs through the
fast Triton kernels, as it does on NVIDIA, with NVIDIA results unchanged.

> **Status: v0.1 in progress, running on RX 9000 (gfx1200). MI300X is planned for v0.2.** On an RX 9060 XT
> (ROCm 10.1) the default `Emeans(d, k).train(x)` uses Triton, every FP32/FP16/BF16 path trains,
> the e-means test suite passes (one known upstream failure, below), the rewritten kernels match
> PyTorch bit for bit, and Triton is 2.2× to 4× faster than PyTorch for BF16 and FP16.
> gfx942 (MI300X) has not been run yet; gfx950 and gfx11xx are compile-checked only.

## Target GPUs

| Family | Architecture | Example GPUs | Plan |
|---|---|---|---|
| CDNA 3 | gfx942 | MI300X, MI325X | compile-checked; validation and tuning in v0.2 |
| CDNA 4 | gfx950 | MI350 series | compile-checked |
| RDNA 3 | gfx11xx | RX 7000 series | compile-checked |
| RDNA 4 | gfx120x | RX 9000 series | **runs**, tested on RX 9060 XT (v0.1) |

Sibling chips (gfx1101, gfx1201, ...) are matched by family. Other AMD GPUs fall back to native
FP32 through PyTorch, which upstream already supports.

## What this fork changes

Upstream e-means supports AMD only as native FP32 through plain PyTorch; FP16/BF16 (including
the default FP32 → BF16 conversion) raise an error, and every GPU test is skipped on ROCm.

| Area | Change |
|---|---|
| Kernels | The 10 PTX inline-assembly snippets in the EMA-update, split and FP8-cast kernels are replaced by plain Triton arithmetic. With fusion off it compiles to the same `.rn` instructions on NVIDIA, and to valid code on all four AMD families. |
| Device detection | New `device_profile.py`: tells NVIDIA from AMD (ROCm reports both as `"cuda"`) and maps AMD chips to a family with its wavefront size, LDS budget and matrix-instruction support. |
| Backend selection | `assignment_backend="auto"` picks Triton for FP32, FP16 and BF16 on the four families, so the default BF16 path works. The PyTorch backend also handles FP16/BF16 there (`torch.addmm(out_dtype=)`), as explicit `native` or as the fallback without Triton. CuTe and TMA descriptors are never selected on AMD. |
| Graph replay | Training uses the eager loop on ROCm (same results, only speed differs). Replay works on gfx1200 but gave no speed-up there, so `EMEANS_ROCM_GRAPH_REPLAY=1` opts in. |
| Errors | FP8 and INT8 on AMD raise an error that says support is planned. |
| Tests | Vendor-aware skips replace the blanket ROCm skips; new GPU-free compile tests and bit-level Triton-vs-PyTorch checks. |

Supported data per GPU is in the kernel table of the
[e-means README](contrib/e_means/README.md#kernels).

## Quick start

Install a ROCm build of PyTorch and Triton in a fresh virtual environment. For an RX 9000 card
(gfx1200) on ROCm 10.1:

```bash
python3 -m venv ~/venvs/emeans-rocm && source ~/venvs/emeans-rocm/bin/activate
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1200]==2.14.0+rocm10.1.0"     # brings triton 3.8.0+rocm10.1.0
python -m pip install faiss-cpu numpy
```

Then make Python use this repository's e-means instead of the copy bundled in the faiss wheel
(see Development setup), and train:

```python
import torch
from faiss.contrib.e_means import Emeans

x = torch.randn(200_000, 64, device="cuda")      # "cuda" also means AMD on ROCm
em = Emeans(64, 1024)
centroids, labels = em.train(x)
print(em.recipe["assignment_backend"])           # expected: "triton" on a supported AMD GPU
```

## Development setup

The `faiss-cpu` wheel ships its own, older copy of `faiss/contrib/e_means`, which would shadow
this repository. Replace it with a symlink to the working tree (the wheel's copy is kept aside):

```bash
source ~/venvs/emeans-rocm/bin/activate
C=$(python -c "import faiss.contrib, os; print(os.path.dirname(faiss.contrib.__file__))")
mv "$C/e_means" "$C/e_means.pypi-orig"
ln -s "$PWD/contrib/e_means" "$C/e_means"
python -c "import faiss.contrib.e_means as m; print(m.__file__)"   # should point here
```

Recorded development stack: ROCm 10.1.0, Python 3.14, torch 2.14.0+rocm10.1.0,
triton 3.8.0+rocm10.1.0, faiss-cpu 1.16.0, on an RX 9060 XT (gfx1200, 32 CUs, wave32, 64 KiB
LDS per workgroup, 16 GB). PyTorch reports `multi_processor_count = 16` there: RDNA counts
work-group processors (two CUs each).

## Running the tests

```bash
cd tests
# Everything except the GPU tests, without touching the GPU:
ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -m pytest -q $(ls *e_means*.py | grep -v -e utils -e conditions)
# Full run on a GPU (add EMEANS_ROCM_GRAPH_REPLAY=1 to include the graph-replay tests):
python -m pytest -q $(ls *e_means*.py | grep -v -e utils -e conditions)
```

- Hide the GPU with `ROCR_VISIBLE_DEVICES=` (empty). `HIP_VISIBLE_DEVICES=-1` makes ROCm 10.1
  abort at the first `torch.cuda.is_available()`.
- GPU-free checks worth knowing: `torch_test_e_means_ptx_free.py` (no inline assembly, no FMA
  contraction on four AMD targets plus sm90), `torch_test_e_means_amd_fences.py` (NVIDIA-only
  features stay off on AMD), `torch_test_e_means_backend_selection.py` and
  `torch_test_e_means_device_profile.py` (mocked devices).
- On the GPU, `torch_test_e_means_triton_numerics.py` checks the rewritten kernels bit for bit
  against PyTorch and correctly rounded references.
- Known failure: `torch_test_e_means_ivf.py::TestIVFFlat::test_all_probes_split_tiles_empty_lists_and_ties`
  fails by one rounding step on this torch build, also on the unmodified upstream snapshot.

GPU smoke matrix (every precision × backend, each case in its own process with a timeout):

```bash
python tools/w0_smoke.py --rows 20000 --k 256          # small, safe on a shared GPU
python tools/w0_smoke.py                               # 200k x 64, k = 1024
EMEANS_ROCM_GRAPH_REPLAY=1 python tools/w0_smoke.py    # same with graph replay
```

`tools/rocm_probe/` holds the offline compile probes (no GPU needed) used to verify the kernel
rewrite; see its README.

## Roadmap

| Version | Scope | Hardware |
|---|---|---|
| **v0.1** (in progress) | Triton as the default AMD path for FP32, FP16 and BF16; validated and tuned on RDNA4 | local RX 9060 XT |
| v0.2 | MI300X (gfx942) validation and tuning; INT8 on all four families; measure whether a hand-written AMD kernel (the CuTe counterpart) is worth building | MI300X (cloud) and RX 9060 XT |
| v0.3 | FP8: native on gfx120x (tested) and gfx950 (compile-checked), optionally emulated on gfx942 and gfx11xx; the hand-written kernel if v0.2's measurement calls for it | RX 9060 XT and MI300X |
| v0.4 | Several GPUs (`DistributedEmeans`, sharded centroids) | 8× MI300X (cloud) |

FP8 on RDNA4 needs only the local card, so it may move into v0.2 if a cloud session slips.

**v0.1 status**

| Item | Status |
|---|---|
| Environment, offline compile checks, device facts | done (gfx1200) |
| PTX inline assembly replaced by plain Triton | done; bit-equal to PyTorch on gfx1200 |
| Device profile and backend selection | done; Triton selected by default on gfx1200 |
| NVIDIA-only Triton features fenced off on AMD | done |
| Graph-replay guard (eager by default on ROCm) | done; replay verified on gfx1200, no speed-up |
| Native FP16/BF16 PyTorch fallback on ROCm | done; verified on gfx1200 |
| Vendor-aware test skips and bit-level numerics tests | done; passing on gfx1200 |
| RX 9060 XT runs | done |
| RDNA4 tuning table (tiles, warps, stages) | in progress: assignment kernel 1.1× to 2.9× faster in the sweep |
| Documentation | done; updated with each item |

Acceptance for v0.1: on gfx1200, the default `Emeans(d, k).train(x)` uses Triton; final loss
within noise of the PyTorch backend; EMA update and split bit-equal to PyTorch; Triton faster
than PyTorch for BF16; NVIDIA behaviour unchanged (selection and kernels are checked with
mocked devices and sm90 compiles; a run on NVIDIA hardware is still outstanding).

**What v0.1 does not cover.** gfx942 (MI300), gfx950 (MI350) and gfx11xx (RX 7000) select the
same Triton path and compile cleanly, but have not been run and use the upstream (NVIDIA-fitted)
tile settings: expected to work, not validated and not tuned. MI300 is covered by v0.2. INT8
and FP8 are refused with an error until v0.2 and v0.3. Several GPUs are expected to work but are
untested until v0.4.

## Repository layout

```
contrib/e_means/     e-means source (upstream + ROCm changes); its README is the user guide
benchs/e_means/      upstream benchmarks and demos
tests/               e-means tests (upstream + new ROCm tests)
tools/rocm_probe/    offline Triton compile probes for AMD targets
tools/w0_smoke.py    precision x backend smoke matrix
```

## Relation to upstream faiss

This repository starts from [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
`main` at commit `14392c952788ca1d6980361cf01d1c8736afc4db` (2026-11-18). The first commit is an
unmodified snapshot of `contrib/e_means`, `benchs/e_means` and the e-means tests, so
`git diff 2cb8b5f` shows exactly what this fork changes. Paths mirror faiss, so commits apply
to a faiss checkout with `git format-patch` / `git am`. To compare with a newer upstream:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss
git -C /tmp/faiss sparse-checkout set contrib/e_means benchs/e_means tests
diff -r /tmp/faiss/contrib/e_means contrib/e_means
```

## License

MIT, as faiss; see [LICENSE](LICENSE). The e-means code is by Meta Platforms, Inc. and
affiliates; the ROCm changes in this repository are under the same license.
