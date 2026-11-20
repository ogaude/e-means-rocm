# e-means-rocm

AMD GPU (ROCm) support for [e-means](contrib/e_means/README.md), the mini-batch k-means in
faiss `contrib/e_means`. The goal is that `Emeans(d, k).train(x)` runs on AMD GPUs through the
fast Triton kernels, as it does on NVIDIA, with NVIDIA results unchanged.

> **Status: v0.2 released (tag `e-means-rocm-v0.2`) for RX 9000 (gfx1200). MI300X follows in v0.3.**
> On an RX 9060 XT (gfx1200, ROCm 10.1) the default `Emeans(d, k).train(x)` uses Triton, every
> data format trains (FP32, FP16, BF16, INT8/UINT8, FP8), the whole e-means test suite passes,
> and the rewritten kernels match PyTorch bit for bit. Triton is 2.1× to 9.3× faster than
> PyTorch for BF16, and tuned INT8 and FP8 train 1.02× to 1.37× faster than BF16. MI300X
> (gfx942) is planned for v0.3; MI350 (gfx950) and RX 7000 (gfx11xx) are compile-checked only.
> **NVIDIA has not been re-run on hardware** (see Known limitations).

## Quick start

Clone this repository and install a ROCm build of PyTorch and Triton in a fresh virtual
environment. For an RX 9000 card (gfx1200) on ROCm 10.1:

```bash
git clone https://github.com/ogaude/e-means-rocm && cd e-means-rocm
python3 -m venv ~/venvs/emeans-rocm && source ~/venvs/emeans-rocm/bin/activate
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1200]==2.14.0+rocm10.1.0"     # brings triton 3.8.0+rocm10.1.0
python -m pip install faiss-cpu numpy
```

The wheel extra is named after the GPU target (`device-gfx1200` here; `rocminfo` shows yours).
Inside a clone, `tools/gpu_session/setup.sh` does the install and the next step for the
detected target.

The `faiss-cpu` wheel ships its own, older copy of `faiss/contrib/e_means`, which would shadow
this repository. From the repository root, replace it with a symlink to the working tree (the
wheel's copy is kept aside):

```bash
C=$(python -c "import faiss.contrib, os; print(os.path.dirname(faiss.contrib.__file__))")
mv "$C/e_means" "$C/e_means.pypi-orig"
ln -s "$PWD/contrib/e_means" "$C/e_means"
python -c "import os, faiss.contrib.e_means as m; print(os.path.realpath(m.__file__))"  # this repo
```

Then train:

```python
import torch
from faiss.contrib.e_means import Emeans

x = torch.randn(200_000, 64, device="cuda")      # "cuda" also means AMD on ROCm
em = Emeans(64, 1024)
centroids, labels = em.train(x)
print(em.recipe["assignment_backend"])           # expected: "triton" on a supported AMD GPU
```

INT8 and FP8 data go through `QuantizedEmeans` with an `Int8Quantizer` or `Fp8Quantizer`; see
the [e-means README](contrib/e_means/README.md#kernels).

## Target GPUs

| Family | Architecture | Example GPUs | FP32 / FP16 / BF16 | INT8 | FP8 (E4M3) |
|---|---|---|---|---|---|
| CDNA 3 | gfx942 | MI300X, MI325X | compile-checked; run and tuning in v0.3 (cloud) | compile-checked | refused (no native E4M3) |
| CDNA 4 | gfx950 | MI350 series | compile-checked | compile-checked | compile-checked |
| RDNA 3 | gfx11xx | RX 7000 series | compile-checked | compile-checked | refused (no native E4M3) |
| RDNA 4 | gfx120x | RX 9000 series | **runs, tuned** (v0.1) | **runs, tuned** (v0.2) | **runs, tuned** (v0.2) |

"Compile-checked" means the kernels compile for that target within its shared-memory (LDS)
budget and are selected automatically, but have not been run. They use the NVIDIA-fitted tile
settings, except that INT8 gets a pipeline depth that fits the family's LDS. Sibling chips
(gfx1101, gfx1201, ...) are matched by family. Other AMD GPUs fall back to native FP32 through
PyTorch, which upstream already supports.

## Results on RX 9000 (gfx1200)

Measured on an RX 9060 XT (16 GB), ROCm 10.1, torch 2.14.0+rocm10.1.0, triton 3.8.0, all in one
run (2026-11-20). Times are warm `train()` runs of 3 epochs on random FP32 data, converted to
each format by the default path or a calibrated quantizer; "PyTorch" is
`assignment_backend="native"`. `tools/bench_train.py` reproduces these runs.

| Data format | 200k × 64, 1,024 centroids | 1M × 128, 4,096 centroids | 4M × 128, 16,384 centroids |
|---|---|---|---|
| BF16, Triton (the default for FP32 input) | 0.028 s | 0.232 s | 1.491 s |
| BF16, PyTorch | 0.058 s | 1.008 s | 13.864 s |
| FP16, Triton | 0.029 s | 0.227 s | 1.503 s |
| FP16, PyTorch | 0.058 s | 0.980 s | 13.410 s |
| Native FP32, Triton | 0.039 s | 0.638 s | 8.601 s |
| Native FP32, PyTorch | 0.063 s | 1.015 s | 14.132 s |
| INT8, Triton | 0.028 s | 0.193 s | 1.090 s |
| FP8, Triton | **0.025 s** | **0.169 s** | 1.143 s |

| Speed-up | 200k × 64 | 1M × 128 | 4M × 128 |
|---|---|---|---|
| Triton over PyTorch, BF16 | 2.1× | 4.3× | 9.3× |
| Triton over PyTorch, FP16 | 2.0× | 4.3× | 8.9× |
| Triton over PyTorch, FP32 | 1.6× | 1.6× | 1.6× |
| INT8 over BF16 (Triton) | 1.02× | 1.20× | 1.37× |
| FP8 over BF16 (Triton) | 1.13× | 1.37× | 1.30× |

Final losses: FP16 and FP32 within 0.02% of BF16, INT8 within 0.06%, FP8 within 0.08%.

### What the RDNA4 tuning gains

Assignment kernel, tuned tables against the NVIDIA-fitted settings (`tools/tune_assign.py
--compare`, d = 32 to 768, k = 1,024 to 16,384, re-measured 2026-11-20). Labels are the same,
or the winning distances agree within FP32 rounding where ties are ordered differently; INT8
results are identical.

| Format | 65,536-row batches | 2,048-row batches |
|---|---|---|
| BF16 / FP16 | 1.06× to 2.80× | 0.95× to 3.45× |
| Native FP32 | 1.16× to 1.40× | 1.04× to 1.34× |
| INT8 | 2.99× to 8.51× | 2.57× to 10.4× |
| FP8 | 1.22× to 2.74× | 0.99× to 2.75× |

Training, each format with its tables on against off (same run, 3 warm epochs):

| Format | 200k × 64 | 1M × 128 | 4M × 128 |
|---|---|---|---|
| BF16 | 1.00× | 1.26× | 1.80× |
| INT8 | 1.20× | 1.35× | 1.94× |
| FP8 | 1.08× | 1.31× | 1.66× |

The split-K choice (below) does not affect training, whose batches are large.

### Findings

- **Keep BF16 on RX cards.** RDNA has no FP32 matrix instruction, so native FP32 costs 1.4× to
  5.8× the time of BF16 here, more as the problem grows. Triton is still the faster FP32 path,
  so it stays the automatic choice.
- **The NVIDIA-fitted tile settings are slow on RDNA4** (wave32): mostly from 128-row tiles at
  d > 128, and the stage count and tile shape on the full-row path (64 < d ≤ 128).
- **INT8 and FP8 need their own settings even more.** With the NVIDIA-fitted settings for both,
  INT8 trained 0.83× to 1.27× as fast as BF16; with the tuned ones for both, 1.00× to 1.37×
  (same run as the tuning-gain table). The largest INT8 kernel gains
  (5× to 10×) are at d ≤ 64, where upstream pads every row to 128. One pipeline stage won on
  every INT8 and FP8 shape swept.
- **Settings tuned at one batch size can lose at another.** The v0.1 BF16 settings for d ≤ 64,
  tuned at 65,536 rows, ranged from 0.55× to 1.72× of upstream's speed between 2,048 and
  262,144 rows (128-row tiles at d ≤ 32 leave work-groups idle on small batches). One
  configuration chosen over that whole range now stays at 0.96× to 1.69×.
- **Split-K pays only on very small row counts on RDNA4.** At 256 rows it is 1.1× to 3.9× faster
  in all but one case (d = 64 with 4,096 centroids: 0.71×, still chosen). At 4,096 rows it is
  2.4× to 3.8× slower, yet the NVIDIA-fitted rule still chose it there, so RDNA4 now uses it up
  to 512 rows only (`tools/tune_split_k.py`). The cap also passes on a 1.02× to 1.14× gain at
  1,024 rows with d = 128.
- **FP8 encodes exactly like PyTorch.** Every FP32 value up to 448 in magnitude (2.28 billion)
  gives the same E4M3 code through Triton's conversion as through
  `tensor.to(torch.float8_e4m3fn)`.
- **Graph replay brings no speed-up on this card** (0.94× and 1.00×), so training runs eagerly
  by default on ROCm; replay is verified and available with `EMEANS_ROCM_GRAPH_REPLAY=1`.
- **Pitfalls for AMD code in general:** on ROCm `torch.cuda.get_device_capability()` returns gfx
  numbers ((12, 0) here), so a bare `>= (8, 0)` check silently matches AMD;
  `multi_processor_count` is 16 here because RDNA counts work-group processors (two CUs each);
  and `HIP_VISIBLE_DEVICES=-1` aborts ROCm 10.1 (use `ROCR_VISIBLE_DEVICES=` to hide the GPU).

## What this fork changes

Upstream e-means does not use its GPU kernels on AMD. Automatic selection treats ROCm as having
no fast backend, so by default only native FP32 through plain PyTorch works, and the default
FP32 → BF16 conversion makes `Emeans(d, k).train(x)` raise an error; native FP32 needs
`QuantizedEmeans(d, k, quantizer=None)`. An explicit `assignment_backend="triton"` is not
refused, but upstream's NVIDIA inline assembly does not run correctly on AMD. Most GPU tests
(Triton, FP16/BF16, INT8, FP8) are skipped on ROCm.

| Area | Change |
|---|---|
| Kernels | The 10 PTX inline-assembly snippets in the EMA-update, split and FP8-cast kernels are replaced by plain Triton arithmetic. With fusion off it compiles to the same `.rn` instructions on NVIDIA, and to valid code on all four AMD families. |
| Device detection | New `device_profile.py`: tells NVIDIA from AMD (ROCm reports both as `"cuda"`) and maps AMD chips to a family with its wavefront size, LDS budget and matrix-instruction support. |
| Backend selection | `assignment_backend="auto"` picks Triton for FP32, FP16 and BF16 on the four families, so the default BF16 path works. The PyTorch backend also handles FP16/BF16 there (`torch.addmm(out_dtype=)`), as explicit `native` or as the fallback without Triton. CuTe and TMA descriptors are never selected on AMD. |
| INT8 (v0.2) | INT8/UINT8 assignment on the four families' integer matrix instructions, with a per-family pipeline depth that fits in 64 KiB of LDS (upstream's three stages need up to 112 KiB outside gfx942). |
| FP8 (v0.2) | FP8 (E4M3) assignment on the families with a native E4M3 matrix instruction, gfx950 and gfx120x. The scaled split's NVIDIA-only `libdevice.ldexp` is replaced on AMD by an exact power-of-two multiply (bit-identical to `torch.ldexp`). gfx942 and gfx11xx refuse FP8 with an error. |
| Tuning | Per-family settings in `triton_assignment.py`, tuned on RDNA4: `_AMD_DIRECT_CONFIGS` (FP32, FP16/BF16, FP8), `_AMD_INT8_CONFIGS` with `_AMD_INT8_SMALL_CONFIGS` (batches up to 4,096 rows; 2,048 at d = 256), and `_AMD_SPLIT_K_MAX_ROW_BLOCKS`. Other families keep the NVIDIA-fitted settings. Gains are under Results. |
| Graph replay | Training uses the eager loop on ROCm (same results, only speed differs); `EMEANS_ROCM_GRAPH_REPLAY=1` opts in to replay. |
| Tests | Vendor-aware skips replace the ROCm skips; INT8 and FP8 tests run on AMD; new GPU-free compile tests (no inline assembly, NVIDIA-only features fenced off, INT8 LDS fit on four AMD targets) and bit-level checks against PyTorch, the FP8 encode included. One upstream IVF test made robust to OpenBLAS builds. |

Supported data per GPU and the converters are in the kernel table of the
[e-means README](contrib/e_means/README.md#kernels).

## Known limitations

- **Only RX 9000 (gfx120x) has been run.** MI300 (gfx942), MI350 (gfx950) and RX 7000 (gfx11xx)
  compile and are selected automatically, but are not validated and not tuned. MI300X is
  planned for v0.3.
- **FP8 runs only where E4M3 is native** (gfx120x, gfx950). MI300 and RX 7000 refuse it;
  emulated FP8 there is a possible v0.4 item.
- **NVIDIA has not been re-run on hardware.** "NVIDIA results unchanged" rests on mocked-device
  tests of the backend selection and on sm90 compiles of the rewritten kernels.
- **Several GPUs** (`DistributedEmeans`, sharded centroids) are expected to work on AMD but are
  untested until v0.5; their tests stay skipped on ROCm.
- **Training is not bit-reproducible by default**, on AMD as upstream on NVIDIA: batch
  statistics are summed with atomic additions, so two identical FP32 runs can end in different
  clusterings. `torch.use_deterministic_algorithms(True)` makes runs identical, eager and with
  graph replay.

## Development

Recorded development stack: ROCm 10.1.0, Python 3.14, torch 2.14.0+rocm10.1.0,
triton 3.8.0+rocm10.1.0, faiss-cpu 1.16.0, on an RX 9060 XT (gfx1200, 32 CUs, wave32, 64 KiB
LDS per workgroup, 16 GB).

### Running the tests

The tests need `python -m pip install pytest`.

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
  features stay off on AMD; RDNA4 table shape and LDS fit), `torch_test_e_means_int8_compile.py`
  (INT8 LDS fit on four AMD targets), `torch_test_e_means_backend_selection.py` and
  `torch_test_e_means_device_profile.py` (mocked devices).
- On the GPU, `torch_test_e_means_triton_numerics.py` checks the rewritten kernels bit for bit
  against PyTorch and correctly rounded references, and `torch_test_e_means_fp8_amd.py` checks
  the FP8 encode and the `ldexp` replacement against PyTorch.
- `torch_test_e_means_ivf.py::TestIVFFlat::test_all_probes_split_tiles_empty_lists_and_ties`
  failed by one rounding step on this torch build, also on the unmodified upstream snapshot:
  the ROCm wheels use OpenBLAS, whose FP32 CPU matmul results depend on the matrix shape, so the
  index's batched products and the test's single product could differ in the last bit. The
  test now requires identical ids and scores within FP32 rounding of the represented operands,
  which still catches unrepresented (FP32) operands by a factor of about 100.

### Tools

```bash
# Smoke matrix: every precision x backend, each case in its own process with a timeout
python tools/w0_smoke.py --rows 20000 --k 256          # small, safe on a shared GPU
python tools/w0_smoke.py                               # 200k x 64, k = 1024
EMEANS_ROCM_GRAPH_REPLAY=1 python tools/w0_smoke.py    # same with graph replay

# Training benchmark (the Results tables)
python tools/bench_train.py --shapes 1000000x128x4096 --precisions bf16 fp16 fp32 int8 fp8 --replay 0

# Assignment-kernel tuning: sweep tiles, warps and stages, each candidate checked against the
# NVIDIA-fitted configuration's results; --compare times the current table against it
python tools/tune_assign.py --quick --k 1024 16384     # BF16/FP16/FP32 sweep, about two hours
python tools/tune_assign.py --dtypes int8 --quick      # INT8 sweep, about 45 minutes
python tools/tune_assign.py --dtypes fp8 --quick --d 64 128 256 768   # FP8 (d % 64 == 0)
python tools/tune_assign.py --compare                  # add --rows 2048 for small batches
python tools/tune_split_k.py                           # direct vs split-K on small row counts
```

Sweep times are for gfx1200. `tools/rocm_probe/` holds the offline compile probes (no GPU
needed) used to verify the kernel rewrite; `tools/gpu_session/` runs the whole validation and
tuning sequence on a fresh cloud GPU; see their READMEs.

## Roadmap

| Version | Scope | Hardware | Status |
|---|---|---|---|
| **v0.1** | Triton as the default AMD path for FP32, FP16 and BF16; validated and tuned on RDNA4 | RX 9060 XT | **released** 2026-11-19 |
| **v0.2** | INT8 on all four families; native FP8 on gfx120x and gfx950; RDNA4 tuning for both, small-batch settings and the split-K choice; the cloud session tooling | RX 9060 XT | **released** 2026-11-20 |
| v0.3 | MI300X (gfx942) validation and tuning; measure whether a hand-written AMD kernel (the CuTe counterpart) is worth building | MI300X (cloud) | planned; scripts ready |
| v0.4 | Optionally emulated FP8 on gfx942 and gfx11xx; the hand-written kernel if v0.3's measurement calls for it | RX 9060 XT and MI300X | planned |
| v0.5 | Several GPUs (`DistributedEmeans`, sharded centroids) | 8× MI300X (cloud) | planned |

The MI300X work was first planned for v0.2; it moved to v0.3 (2026-11-20) so the RX 9000 work
could be released without waiting for cloud access, and the later versions moved up by one.

**v0.3 plan (needs an MI300X)**

| Item | Plan |
|---|---|
| gfx942 validation | smoke matrix, test suite (eager and replay), INT8 there |
| gfx942 tuning | `cdna3` entries for the BF16, INT8 and split-K tables; graph-replay default decided on measured speed |
| Hand-written backend gate | Triton BF16 assignment against rocBLAS/hipBLASLt GEMM (`tools/gemm_gate.py`); build a FlyDSL backend (v0.4) only if Triton reaches less than about 80% of the GEMM's speed |

### Release history

**v0.2 (2026-11-20, tag `e-means-rocm-v0.2`)**, all on RX 9000 (gfx1200):

| Item | Result |
|---|---|
| INT8/UINT8 on the four AMD families | runs on gfx1200, distances exact against a float64 reference; LDS fit compile-checked on the other three |
| FP8 (E4M3) on gfx120x and gfx950 | runs on gfx1200; encode identical to PyTorch for all FP32 values up to 448; `libdevice.ldexp` replaced by an exact power-of-two multiply |
| RDNA4 tuning | INT8 and FP8 tables; BF16/FP16 d ≤ 64 re-chosen for small batches; split-K capped at 512 rows |
| Tests | 253 passed eager, 257 with graph replay, none failing |
| Cloud session tooling | `tools/gpu_session/`, `tools/gemm_gate.py`, `tools/bench_train.py`, `tools/tune_split_k.py`, `tools/device_facts.py`, INT8/FP8 modes in `tools/tune_assign.py` |

**v0.1 (2026-11-19, tag `e-means-rocm-v0.1`)**: PTX inline assembly replaced by plain Triton,
device profile and backend selection, NVIDIA-only Triton features fenced off on AMD, eager
training by default on ROCm, native FP16/BF16 PyTorch fallback, vendor-aware tests, RDNA4
tuning table. Acceptance on gfx1200, as measured then:

| Criterion | Result |
|---|---|
| Default `Emeans(d, k).train(x)` uses Triton | met (`recipe["assignment_backend"] == "triton"`, BF16) |
| Final loss within noise of the PyTorch backend | met (BF16 objectives agree to about 1e-10 relative) |
| EMA update and split bit-equal to PyTorch | met (`torch_test_e_means_triton_numerics.py`) |
| Triton faster than PyTorch for BF16 | met (2.1× to 4.8× at the sizes measured then; 9.3× at 4M × 128 in v0.2) |
| NVIDIA behaviour unchanged | **partly**: selection checked with mocked devices, kernels compiled for sm90 and checked; no run on NVIDIA hardware |

## Repository layout

```
contrib/e_means/      e-means source (upstream + ROCm changes); its README is the user guide
benchs/e_means/       upstream benchmarks and demos
tests/                e-means tests (upstream + new ROCm tests)
tools/rocm_probe/     offline Triton compile probes for AMD targets
tools/w0_smoke.py     precision x backend smoke matrix
tools/bench_train.py  warm train() benchmark: formats, Triton vs PyTorch, eager vs graph replay
tools/tune_assign.py  assignment-kernel tuning sweep and upstream-vs-table comparison
tools/tune_split_k.py direct vs split-K assignment on small row grids
tools/gemm_gate.py    Triton BF16 assignment vs a bare BLAS GEMM (hand-written backend gate)
tools/device_facts.py GPU and stack facts as JSON
tools/gpu_session/    setup + one-shot session runner for a cloud GPU VM (see its README)
```

## Relation to upstream faiss

This repository starts from [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
`main` at commit `14392c952788ca1d6980361cf01d1c8736afc4db` (2026-11-18). The first commit is an
unmodified snapshot of `contrib/e_means`, `benchs/e_means` and the e-means tests, so
`git diff d6e4215 -- contrib benchs tests` shows exactly what this fork changes in faiss's
files. Those paths mirror faiss, so changes to them apply to a faiss checkout
(`git format-patch d6e4215 -- contrib benchs tests`, then `git am`); `README.md`, `tools/` and
`.gitignore` belong to this fork only. To see what upstream changed since the snapshot:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss
git -C /tmp/faiss sparse-checkout set contrib/e_means benchs/e_means tests
mkdir -p /tmp/snapshot && git archive d6e4215 contrib/e_means | tar -x -C /tmp/snapshot
diff -r /tmp/snapshot/contrib/e_means /tmp/faiss/contrib/e_means
```

## License

MIT, as faiss; see [LICENSE](LICENSE). The e-means code is by Meta Platforms, Inc. and
affiliates; the ROCm changes in this repository are under the same license.
