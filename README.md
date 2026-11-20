# e-means-rocm

AMD GPU (ROCm) support for [e-means](contrib/e_means/README.md), the mini-batch k-means in
faiss `contrib/e_means`. The goal is that `Emeans(d, k).train(x)` runs on AMD GPUs through the
fast Triton kernels, as it does on NVIDIA, with NVIDIA results unchanged.

> **Status: v0.1 released (tag `e-means-rocm-v0.1`); v0.2 done on RX 9000, MI300X pending.**
> On an RX 9060 XT (gfx1200, ROCm 10.1) the default `Emeans(d, k).train(x)` uses Triton, every
> data format trains (FP32, FP16, BF16, INT8/UINT8, FP8), the whole e-means test suite passes,
> and the rewritten kernels match PyTorch bit for bit. Triton is 2.1× to 9.3× faster than
> PyTorch for BF16, and tuned FP8 and INT8 train 1.0× to 1.4× faster than BF16. gfx942 (MI300X)
> waits for a cloud session; gfx950 and gfx11xx are compile-checked only. **NVIDIA has not been
> re-run on hardware:** "NVIDIA unchanged" rests on mocked-device tests and sm90 compiles (see
> v0.1 acceptance).

## Target GPUs

| Family | Architecture | Example GPUs | FP32 / FP16 / BF16 | INT8 | FP8 (E4M3) |
|---|---|---|---|---|---|
| CDNA 3 | gfx942 | MI300X, MI325X | compile-checked; run and tuning in v0.2 (cloud) | compile-checked | refused (no native E4M3) |
| CDNA 4 | gfx950 | MI350 series | compile-checked | compile-checked | compile-checked |
| RDNA 3 | gfx11xx | RX 7000 series | compile-checked | compile-checked | refused (no native E4M3) |
| RDNA 4 | gfx120x | RX 9000 series | **runs, tuned** (v0.1) | **runs, tuned** (v0.2) | **runs, tuned** (v0.2) |

"Compile-checked" means the kernels compile for that target within its shared-memory budget and
are selected automatically, with the NVIDIA-fitted tile settings; they have not been run.
Sibling chips (gfx1101, gfx1201, ...) are matched by family. Other AMD GPUs fall back to native
FP32 through PyTorch, which upstream already supports.

## What this fork changes

Upstream e-means supports AMD only as native FP32 through plain PyTorch; FP16/BF16 (including
the default FP32 → BF16 conversion) raise an error, and every GPU test is skipped on ROCm.

| Area | Change |
|---|---|
| Kernels | The 10 PTX inline-assembly snippets in the EMA-update, split and FP8-cast kernels are replaced by plain Triton arithmetic. With fusion off it compiles to the same `.rn` instructions on NVIDIA, and to valid code on all four AMD families. |
| Tuning | Per-family assignment-kernel settings in `triton_assignment.py`, all tuned on RDNA4 with identical results. FP32/FP16/BF16 (`_AMD_DIRECT_CONFIGS`): kernel up to 2.7× faster (1.0× to 2.7× over 81 shapes, none slower), training 1.25× to 1.35× faster; the d ≤ 64 rule was re-chosen in v0.2 to hold up for small batches too. INT8 (`_AMD_INT8_CONFIGS`, plus `_AMD_INT8_SMALL_CONFIGS` for batches up to 4,096 rows): kernel 2.4× to 10× faster. FP8 (`"fp8"` rules in `_AMD_DIRECT_CONFIGS`): kernel 1.0× to 2.9× faster. Split-K (`_AMD_SPLIT_K_MAX_ROW_BLOCKS`): used only for small row counts, where it pays on RDNA4. Other families keep the NVIDIA-fitted constants. |
| Device detection | New `device_profile.py`: tells NVIDIA from AMD (ROCm reports both as `"cuda"`) and maps AMD chips to a family with its wavefront size, LDS budget and matrix-instruction support. |
| Backend selection | `assignment_backend="auto"` picks Triton for FP32, FP16 and BF16 on the four families, so the default BF16 path works. The PyTorch backend also handles FP16/BF16 there (`torch.addmm(out_dtype=)`), as explicit `native` or as the fallback without Triton. CuTe and TMA descriptors are never selected on AMD. |
| Graph replay | Training uses the eager loop on ROCm (same results, only speed differs). Replay works on gfx1200 but gave no speed-up there, so `EMEANS_ROCM_GRAPH_REPLAY=1` opts in. |
| INT8 (v0.2) | INT8/UINT8 assignment runs on the four families' integer matrix instructions. Untuned families get a per-family pipeline depth that fits in 64 KiB of LDS (upstream's three stages need up to 112 KiB outside gfx942). |
| FP8 (v0.2) | FP8 (E4M3) assignment runs on the families with a native E4M3 matrix instruction, gfx950 and gfx120x. The scaled split's NVIDIA-only `libdevice.ldexp` is replaced on AMD by an exact power-of-two multiply (bit-identical to `torch.ldexp`). gfx942 and gfx11xx refuse FP8 with an error. |
| Tests | Vendor-aware skips replace the blanket ROCm skips; INT8 and FP8 tests run on AMD; new GPU-free compile tests (including INT8 LDS fit for all four families) and bit-level Triton-vs-PyTorch checks, FP8 encode included. One upstream IVF test made robust to OpenBLAS builds. |

Supported data per GPU is in the kernel table of the
[e-means README](contrib/e_means/README.md#kernels).

## Results on RX 9000 (gfx1200)

Measured on an RX 9060 XT (16 GB), ROCm 10.1, torch 2.14.0+rocm10.1.0, triton 3.8.0, all in one
run (2026-11-20). Times are warm `train()` runs of 3 epochs on random FP32 data, converted to
each format by the default path or a calibrated quantizer; "PyTorch" is
`assignment_backend="native"`.

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
| Triton over PyTorch, FP32 | 1.6× | 1.6× | 1.6× |
| INT8 over BF16 (Triton) | 1.02× | 1.20× | 1.37× |
| FP8 over BF16 (Triton) | 1.13× | 1.37× | 1.30× |

Final losses: FP16 and FP32 within 0.02% of BF16, INT8 within 0.06%, FP8 within 0.08%.

What the RDNA4 tuning gained (assignment kernel, against the NVIDIA-fitted settings, identical
results; `tools/tune_assign.py --compare`):

| Format | Kernel speed-up | Effect on training |
|---|---|---|
| FP32 / FP16 / BF16 | 1.0× to 2.7× (v0.1); d ≤ 64 re-chosen in v0.2 to hold from 2,048 to 262,144 rows | 1.25× to 1.35× faster (v0.1 measurement) |
| INT8 | 2.4× to 10× | from 0.86× of BF16 to 1.02× to 1.37× |
| FP8 | 1.0× to 2.9× | from 0.79× to 1.08× of BF16 to 1.13× to 1.37× |
| Split-K choice | 1.3× to 3.9× on calls of 1,024 to 8,192 rows | none (large batches never use it) |

Findings:

- **Keep BF16 on RX cards.** RDNA has no FP32 matrix instruction, so native FP32 costs 1.4× to
  5.8× the time of BF16 here, more as the problem grows. Triton is still the faster FP32 path, so it stays the automatic choice.
- **Results are not bit-reproducible by default**, on AMD as upstream on NVIDIA: batch statistics
  are summed with atomic additions, so two identical FP32 runs can end in different clusterings
  (in a BF16 test the labels agreed but centroid bits still differed slightly). `torch.use_deterministic_algorithms(True)` makes runs identical,
  eager and with graph replay.
- **Graph replay brings no speed-up on this card**, so training runs eagerly by default on ROCm;
  replay is verified and available with `EMEANS_ROCM_GRAPH_REPLAY=1`.
- **The NVIDIA-fitted tile settings are slow on RDNA4** (wave32): up to 2.7× in the assignment
  kernel, mostly from wide row tiles at d > 128 and the stage count at d ≤ 128. A sweep and
  comparison tool, `tools/tune_assign.py`, made the RDNA4 table.
- **INT8 and FP8 need their own settings even more.** With the NVIDIA-fitted ones INT8 was 1.16×
  slower than BF16 here and FP8 up to 1.27× slower; tuned, both are faster than BF16. For INT8 the
  largest kernel gain (5× to 10×) is at d ≤ 64, where upstream pads every row to 128. One pipeline
  stage won on nearly every INT8 and FP8 shape.
- **Tile settings tuned at one batch size can lose at another.** The v0.1 BF16 settings for
  d ≤ 64, tuned at 65,536 rows, fell to 0.6× of upstream's speed at 2,048 to 32,768 rows with
  1,024 centroids (128-row tiles leave half the card's work-groups idle on small batches). One
  configuration chosen over 2,048 to 262,144 rows now stays at 0.92× to 1.55×; training is
  unchanged.
- **FP8 encodes exactly like PyTorch on RDNA4.** Every FP32 value up to 448 in magnitude (2.28
  billion) gives the same E4M3 code through Triton's conversion as through
  `tensor.to(torch.float8_e4m3fn)`.
- **Split-K pays only on very small row counts on RDNA4.** It is 1.2× to 3.9× faster at 256 rows
  but 2.4× to 3.9× slower at 4,096 rows, where the NVIDIA-fitted rule still chose it; RDNA4 now
  uses it up to 512 rows (`tools/tune_split_k.py`).
- **Pitfalls for AMD code in general:** on ROCm `torch.cuda.get_device_capability()` returns gfx
  numbers ((12, 0) here), so a bare `>= (8, 0)` check silently matches AMD; `multi_processor_count`
  is 16 here because RDNA counts work-group processors (two CUs each); and
  `HIP_VISIBLE_DEVICES=-1` aborts ROCm 10.1 (use `ROCR_VISIBLE_DEVICES=` to hide the GPU).

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
  features stay off on AMD), `torch_test_e_means_int8_compile.py` (INT8 shared-memory fit on four
  AMD targets), `torch_test_e_means_backend_selection.py` and
  `torch_test_e_means_device_profile.py` (mocked devices).
- On the GPU, `torch_test_e_means_triton_numerics.py` checks the rewritten kernels bit for bit
  against PyTorch and correctly rounded references, and `torch_test_e_means_fp8_amd.py` checks the
  FP8 encode and the `ldexp` replacement against PyTorch.
- `torch_test_e_means_ivf.py::TestIVFFlat::test_all_probes_split_tiles_empty_lists_and_ties`
  used to fail by one rounding step on this torch build (also on the unmodified upstream
  snapshot): the ROCm wheels use OpenBLAS, whose FP32 CPU matmul results depend on the matrix
  shape, so the index's batched products and the test's single product could differ in the last
  bit. The test now requires identical ids and scores within FP32 rounding of the represented
  operands, which still catches unrepresented (FP32) operands by a factor of about 100.

GPU smoke matrix (every precision × backend, each case in its own process with a timeout):

```bash
python tools/w0_smoke.py --rows 20000 --k 256          # small, safe on a shared GPU
python tools/w0_smoke.py                               # 200k x 64, k = 1024
EMEANS_ROCM_GRAPH_REPLAY=1 python tools/w0_smoke.py    # same with graph replay
```

`tools/rocm_probe/` holds the offline compile probes (no GPU needed) used to verify the kernel
rewrite; see its README.

Assignment-kernel tuning (sweep tile sizes, warps and stages; every candidate is checked against
the upstream configuration's results):

```bash
python tools/tune_assign.py --quick --k 1024 16384     # full sweep, about two hours on gfx1200
python tools/tune_assign.py --compare                  # upstream constants vs this GPU's table
python tools/tune_assign.py --dtypes int8 --quick      # INT8 kernel sweep (about 45 min on gfx1200)
python tools/tune_assign.py --dtypes fp8 --quick --d 64 128 256 768   # FP8 sweep (d % 64 == 0)
python tools/tune_split_k.py                           # direct vs split-K on small row counts
```

## Roadmap

| Version | Scope | Hardware | Status |
|---|---|---|---|
| **v0.1** | Triton as the default AMD path for FP32, FP16 and BF16; validated and tuned on RDNA4 | RX 9060 XT | **released** 2026-11-19 |
| **v0.2** | INT8 on all four families; native FP8 on gfx120x and gfx950 (moved from v0.3); RDNA4 tuning for both; MI300X (gfx942) validation and tuning; measure whether a hand-written AMD kernel (the CuTe counterpart) is worth building | RX 9060 XT and MI300X (cloud) | RX 9000 part **done**; MI300X part waiting for a cloud session |
| v0.3 | Optionally emulated FP8 on gfx942 and gfx11xx; the hand-written kernel if v0.2's measurement calls for it | RX 9060 XT and MI300X | planned |
| v0.4 | Several GPUs (`DistributedEmeans`, sharded centroids) | 8× MI300X (cloud) | planned |

**v0.1 status (released 2026-11-19, tag `e-means-rocm-v0.1`)**

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
| RDNA4 tuning table (tiles, warps, stages) | done: assignment kernel up to 2.7× faster, training 1.25× to 1.35× faster on gfx1200 |
| Documentation | done; updated with each item |

Acceptance for v0.1, on gfx1200:

| Criterion | Result |
|---|---|
| Default `Emeans(d, k).train(x)` uses Triton | met (`recipe["assignment_backend"] == "triton"`, BF16) |
| Final loss within noise of the PyTorch backend | met (BF16 objectives agree to about 1e-10 relative) |
| EMA update and split bit-equal to PyTorch | met (`torch_test_e_means_triton_numerics.py`) |
| Triton faster than PyTorch for BF16 | met (2.1× to 4.8×) |
| NVIDIA behaviour unchanged | **partly**: selection checked with mocked devices, kernels compiled for sm90 and checked; no run on NVIDIA hardware |

**v0.2 status**

Done on RX 9000 (gfx1200):

| Item | Result |
|---|---|
| INT8/UINT8 on the four AMD families | runs on gfx1200; distances exact against a float64 reference; compile-checked (LDS fit) on the other three |
| FP8 (E4M3) on gfx120x and gfx950 | runs on gfx1200; encode identical to PyTorch for all 2.28 billion FP32 values up to 448; `libdevice.ldexp` replaced by an exact power-of-two multiply; gfx942 and gfx11xx refuse FP8 |
| RDNA4 INT8 and FP8 tuning | INT8 and FP8 now train faster than BF16 (see Results) |
| RDNA4 small batches | BF16/FP16 d ≤ 64 settings re-chosen (were down to 0.6× of upstream on small batches); INT8 small-batch rules |
| RDNA4 split-K | used only up to 512 rows; the NVIDIA-fitted rule chose it up to 4,096 to 8,192 rows, 2.4× to 3.9× slower there |
| Tests | 253 passed eager, 257 with graph replay, none failing (an upstream IVF test made robust to OpenBLAS builds) |
| Cloud session tooling | `tools/gpu_session/` (setup and one-shot runner), `tools/gemm_gate.py`, `tools/bench_train.py`, `tools/tune_split_k.py`, INT8/FP8 modes in `tools/tune_assign.py` |

Remaining, needs an MI300X:

| Item | Plan |
|---|---|
| gfx942 validation | smoke matrix, test suite (eager and replay), INT8 there |
| gfx942 tuning | `cdna3` entries for the BF16, INT8 and split-K tables; graph-replay default decided on measured speed |
| Hand-written backend gate | Triton BF16 assignment against rocBLAS/hipBLASLt GEMM; build a FlyDSL backend (v0.3) only if Triton trails by more than about 20% |

**Not covered yet.** gfx942 (MI300), gfx950 (MI350) and gfx11xx (RX 7000) select the same
Triton paths (FP32/FP16/BF16 and INT8) and compile cleanly, but have not been run and use the
upstream (NVIDIA-fitted) tile settings: expected to work, not validated and not tuned. MI300 is
covered by v0.2's cloud session. FP8 runs only where E4M3 is native (gfx120x, gfx950); MI300
and RX 7000 refuse it. Several GPUs are
expected to work but are untested until v0.4.

## Repository layout

```
contrib/e_means/      e-means source (upstream + ROCm changes); its README is the user guide
benchs/e_means/       upstream benchmarks and demos
tests/                e-means tests (upstream + new ROCm tests)
tools/rocm_probe/     offline Triton compile probes for AMD targets
tools/w0_smoke.py     precision x backend smoke matrix
tools/tune_assign.py  assignment-kernel tuning sweep and upstream-vs-table comparison
tools/tune_split_k.py direct vs split-K assignment on small row grids
tools/bench_train.py  warm train() benchmark: Triton vs PyTorch, eager vs graph replay
tools/gemm_gate.py    Triton BF16 assignment vs a bare BLAS GEMM (hand-written backend gate)
tools/device_facts.py GPU and stack facts as JSON
tools/gpu_session/    setup + one-shot session runner for a cloud GPU VM (see its README)
```

## Relation to upstream faiss

This repository starts from [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
`main` at commit `14392c952788ca1d6980361cf01d1c8736afc4db` (2026-11-18). The first commit is an
unmodified snapshot of `contrib/e_means`, `benchs/e_means` and the e-means tests, so
`git diff d6e4215` shows exactly what this fork changes. Paths mirror faiss, so commits apply
to a faiss checkout with `git format-patch` / `git am`. To compare with a newer upstream:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss
git -C /tmp/faiss sparse-checkout set contrib/e_means benchs/e_means tests
diff -r /tmp/faiss/contrib/e_means contrib/e_means
```

## License

MIT, as faiss; see [LICENSE](LICENSE). The e-means code is by Meta Platforms, Inc. and
affiliates; the ROCm changes in this repository are under the same license.
