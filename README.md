# e-means-rocm

AMD GPU (ROCm) support for [e-means](contrib/e_means/README.md), the mini-batch k-means in
faiss `contrib/e_means`. The goal is that `Emeans(d, k).train(x)` runs on AMD GPUs through the
fast Triton kernels, as it does on NVIDIA, with NVIDIA results unchanged.

> **Status: work in progress, not yet run on AMD hardware.** All changes so far are verified by
> CPU tests and by compiling the kernels for AMD targets and inspecting the generated code.
> The first GPU runs (RX 9060 XT, then MI300X) are the next step.

## Target GPUs

| Family | Architecture | Example GPUs | Plan |
|---|---|---|---|
| CDNA 3 | gfx942 | MI300X, MI325X | run on hardware |
| CDNA 4 | gfx950 | MI350 series | compile-checked |
| RDNA 3 | gfx11xx | RX 7000 series | compile-checked |
| RDNA 4 | gfx120x | RX 9000 series | run on hardware |

Sibling chips (gfx1101, gfx1201, ...) are matched by family. Other AMD GPUs fall back to native
FP32 through PyTorch, which upstream already supports.

## What this fork changes

Upstream e-means supports AMD only as native FP32 through plain PyTorch; FP16/BF16 (including
the default FP32 → BF16 conversion) raise an error, and every GPU test is skipped on ROCm.

| Area | Change |
|---|---|
| Kernels | The 10 PTX inline-assembly snippets in the EMA-update, split and FP8-cast kernels are replaced by plain Triton arithmetic. With fusion off it compiles to the same `.rn` instructions on NVIDIA, and to valid code on all four AMD families. |
| Device detection | New `device_profile.py`: tells NVIDIA from AMD (ROCm reports both as `"cuda"`) and maps AMD chips to a family with its wavefront size, LDS budget and matrix-instruction support. |
| Backend selection | `assignment_backend="auto"` picks Triton for FP32, FP16 and BF16 on the four families, so the default BF16 path works. CuTe and TMA descriptors are never selected on AMD. |
| Graph replay | Training uses the eager loop on ROCm until CUDA-graph capture is verified (same results, only speed differs); `EMEANS_ROCM_GRAPH_REPLAY=1` opts in. |
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
triton 3.8.0+rocm10.1.0, faiss-cpu 1.16.0, on an RX 9060 XT (gfx1200, 32 CUs, wave32).

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

**v0.1: Triton as the default AMD path for FP32, FP16 and BF16**

| Item | Status |
|---|---|
| Environment and offline compile checks | done; GPU checks pending |
| PTX inline assembly replaced by plain Triton | done (compile-verified) |
| Device profile and backend selection | done |
| NVIDIA-only Triton features fenced off on AMD | done |
| Graph-replay guard (eager by default on ROCm) | done; capture check pending |
| Vendor-aware test skips and bit-level numerics tests | done; first GPU run pending |
| Documentation | done |
| First runs on RX 9060 XT and MI300X | next |
| Per-family tuning (tiles, stages, warps; FP32 Triton vs PyTorch on RX) | needs measurements |
| Native FP16/BF16 PyTorch fallback on ROCm | needs a hardware check |
| Multi-GPU | out of scope ("expected to work, untested") |

Acceptance for v0.1: the default `Emeans(d, k).train(x)` uses Triton on gfx1200 and gfx942;
final loss within noise of the PyTorch backend and of NVIDIA; EMA update and split bit-equal to
PyTorch; Triton faster than PyTorch for BF16; NVIDIA tests unchanged.

**v0.2:** INT8 on all four families; FP8 on gfx950 and gfx120x (native FP8 matrix instructions);
decide by measurement whether a hand-written AMD kernel (the CuTe counterpart) is worth building.

**v0.3 (optional):** emulated FP8 on gfx942 and gfx11xx, multi-GPU validation, hardware runs on
gfx950 and gfx11xx.

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
