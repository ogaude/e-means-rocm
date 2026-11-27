# e-means-rocm

AMD GPU (ROCm) support for [e-means](https://github.com/ogaude/e-means-rocm/blob/main/contrib/e_means/README.md), the mini-batch k-means in
faiss `contrib/e_means`. The goal is that `Emeans(d, k).train(x)` runs on AMD GPUs through the
fast Triton kernels, as it does on NVIDIA, with NVIDIA results unchanged.

> **Status: v0.2.7 released (tag `e-means-rocm-v0.2.7`) for RX 9000 (gfx1200), installable with
> `pip install emeans-rocm`; synced with upstream faiss `b14b3d69`. MI300X follows in v0.3.**
> On an RX 9060 XT (gfx1200, ROCm 10.1) the default `Emeans(d, k).train(x)` uses Triton, every
> data format trains (FP32, FP16, BF16; INT8/UINT8 and FP8 through `QuantizedEmeans`), the whole e-means test suite passes,
> and the rewritten kernels match PyTorch bit for bit in the numerics tests. Triton is 2.5× to 15× faster than
> PyTorch for BF16; BF16 and FP16 are the fastest formats here up to 1M rows (FP8 trains at 0.65× to 0.86× of BF16's
> speed, INT8 at 0.71× to 1.33×, the fastest format at 4M rows since v0.2.7; see Results). MI300X
> (gfx942) is planned for v0.3; MI350 (gfx950) and RX 7000 (gfx11xx) are compile-checked only.
> **NVIDIA has not been re-run on hardware** (see Known limitations).

## Quick start

Install a ROCm build of PyTorch and Triton in a fresh virtual environment, then the package
`emeans-rocm` from PyPI. For an RX 9000 card (gfx1200) on ROCm 10.1:

```bash
python3 -m venv ~/venvs/emeans-rocm && source ~/venvs/emeans-rocm/bin/activate
python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  "torch[device-gfx1200]==2.14.0+rocm10.1.0"     # brings triton 3.8.0+rocm10.1.0
python -m pip install emeans-rocm
```

The wheel extra is named after the GPU target (`device-gfx1200` here; `rocminfo` shows yours).
Install torch first: the package does not list it as a dependency, because pip would then
install the CUDA build from PyPI. The package pulls in `faiss-cpu` and `numpy` (in a conda
environment with `faiss-gpu`, install it with `--no-deps` to keep pip's faiss-cpu out). For the code on
`main` instead of the last release, use `python -m pip install "emeans-rocm @
git+https://github.com/ogaude/e-means-rocm"`, or `python -m pip install .` from a clone.

The package's import name is `emeans_rocm`. It holds the same code as `contrib/e_means` and
sits next to the installed faiss, whose own `faiss.contrib.e_means` (upstream's, older copy)
it leaves alone:

```python
import torch
from emeans_rocm import Emeans

x = torch.randn(200_000, 64, device="cuda")      # "cuda" also means AMD on ROCm
em = Emeans(64, 1024)
centroids, labels = em.train(x)
print(em.recipe["assignment_backend"])           # expected: "triton" on a supported AMD GPU
```

INT8 and FP8 data go through `QuantizedEmeans` with an `Int8Quantizer` or `Fp8Quantizer`; see
the [e-means README](https://github.com/ogaude/e-means-rocm/blob/main/contrib/e_means/README.md#kernels), whose examples import
`faiss.contrib.e_means` (read `emeans_rocm` with the package).

### Development setup

The tests, benchmarks and tools import `faiss.contrib.e_means`, as upstream's do. To work on
them, clone the repository (`git clone https://github.com/ogaude/e-means-rocm && cd
e-means-rocm`), install torch as above and `python -m pip install faiss-cpu numpy`; inside a
clone, `tools/gpu_session/setup.sh` does the install and the next step for the detected target.

The `faiss-cpu` wheel ships its own, older copy of `faiss/contrib/e_means`, which would shadow
this repository. From the repository root, replace it with a symlink to the working tree (the
wheel's copy is kept aside):

```bash
C=$(python -c "import faiss.contrib, os; print(os.path.dirname(faiss.contrib.__file__))")
mv "$C/e_means" "$C/e_means.pypi-orig"
ln -s "$PWD/contrib/e_means" "$C/e_means"
python -c "import os, faiss.contrib.e_means as m; print(os.path.realpath(m.__file__))"  # this repo
```

The example above then runs with `from faiss.contrib.e_means import Emeans`. Remove the
symlink before reinstalling or uninstalling `faiss-cpu`: pip follows it and can delete or
overwrite files in the clone.

`python -m pip install -e .` does not work: the package maps `contrib/e_means` onto
`emeans_rocm` at build time, which an editable install cannot do, so the build stops with an
error. Reinstall with `python -m pip install .` after changes, or work through the symlink.

## Target GPUs

| Family | Architecture | Example GPUs | FP32 / FP16 / BF16 | INT8 | FP8 (E4M3) |
|---|---|---|---|---|---|
| CDNA 3 | gfx942 | MI300X, MI325X | compile-checked; run and tuning in v0.3 (cloud) | compile-checked | refused (no native E4M3) |
| CDNA 4 | gfx950 | MI350 series | compile-checked | compile-checked | compile-checked |
| RDNA 3 | gfx11xx | RX 7000 series, RDNA 3.5 APUs | compile-checked | compile-checked | refused (no native E4M3) |
| RDNA 4 | gfx120x | RX 9000 series | **runs, tuned** (v0.1) | **runs, tuned** (v0.2) | **runs, tuned** (v0.2) |

"Compile-checked" means that every kernel configuration the launcher can pick for that target
compiles within its shared-memory (LDS) budget, checked by GPU-free tests
(`torch_test_e_means_launch_config.py`, `torch_test_e_means_int8_compile.py`), and is selected
automatically, but has not been run. They use the NVIDIA-fitted tile settings, except that INT8
gets a pipeline depth that fits the family's LDS, and gfx950 a 64-row tile and one split-K stage
(see Known limitations). Sibling chips
(gfx1101, gfx1201, ...) are matched by family. Other AMD GPUs fall back to native FP32 through
PyTorch, which upstream already supports.

## Results on RX 9000 (gfx1200)

Measured on an RX 9060 XT (16 GB), ROCm 10.1, torch 2.14.0+rocm10.1.0, triton 3.8.0, on
2026-11-27 (v0.2.7: upstream faiss `b14b3d69`, with cost-based cluster revival as the
default; Triton rows best of three rounds). Times are warm `train()` runs of 3 epochs on random FP32 data, converted to each format:
BF16 and FP16 by the default path during training, INT8 and FP8 by a calibrated quantizer before
timing (encoding not timed, as when data is stored encoded). "PyTorch" is
`assignment_backend="native"`. `tools/bench_train.py` reproduces these runs; speed-ups below are
computed from the unrounded times.

| Data format | 200k × 64, 1,024 centroids | 1M × 128, 4,096 centroids | 4M × 128, 16,384 centroids |
|---|---|---|---|
| BF16, Triton (the default for FP32 input) | 0.036 s | 0.137 s | 1.148 s |
| BF16, PyTorch | 0.089 s | 1.332 s | 17.266 s |
| FP16, Triton | **0.031 s** | **0.118 s** | 1.160 s |
| FP16, PyTorch | 0.090 s | 1.319 s | 16.953 s |
| Native FP32, Triton | 0.044 s | 0.555 s | 7.930 s |
| Native FP32, PyTorch | 0.095 s | 1.369 s | 15.731 s |
| INT8, Triton | 0.041 s | 0.194 s | **0.866 s** |
| FP8, Triton | 0.042 s | 0.211 s | 1.404 s |

| Speed-up | 200k × 64 | 1M × 128 | 4M × 128 |
|---|---|---|---|
| Triton over PyTorch, BF16 | 2.5× | 9.7× | 15.0× |
| Triton over PyTorch, FP16 | 3.0× | 11.2× | 14.6× |
| Triton over PyTorch, FP32 | 2.2× | 2.5× | 2.0× |
| INT8 over BF16 (Triton) | 0.86× | 0.71× | 1.33× |
| FP8 over BF16 (Triton) | 0.86× | 0.65× | 0.82× |

Final losses: FP16 and FP32 within 0.01% of BF16, INT8 within 0.06%, FP8 within 0.08%. The
PyTorch rows are from one round of the same day.

**v0.2.6 compared with v0.2.5** (v0.2.6's run of 2026-11-26 against v0.2.5's tables from 2026-11-20; a run of v0.2.5 on 2026-11-26 reproduced
them within 1.2% at 1M × 128): BF16 and FP16 trained 1.3× to 1.9× faster at the two larger shapes,
and 0.88× (BF16) and 0.95× (FP16) as fast at 200k × 64.
The gain comes from the revival path, which sums each batch in one fused Triton kernel; with
revival off (`unused_threshold=0`), BF16 at 1M × 128 takes 0.235 s, v0.2.5's time. INT8 and FP8 then trained
slower than BF16 (in v0.2.5 they were 1.0× to 1.4× faster), because revival also needs each
row's second-nearest distance and its decoded values. Right after the sync they were 2.2× to
2.6× slower than in v0.2.5 at 1M × 128; v0.2.6 brought that to 1.2× there (1.2× to 1.6×
across the three shapes, the most at 200k × 64) with fused revival statistics, codes decoded in
the kernel, and launch rules for the second-nearest output. At 1M × 128, `unused_threshold=0`
turns revival off and nearly gives back v0.2.5's INT8/FP8 times (v0.2.6: 0.200 s and 0.176 s against
0.193 s and 0.169 s), at a cost in quality: the training objective rose by about 0.9%.

**v0.2.7, INT8.** A stock `Int8Quantizer` now sums its integer codes, exactly, and decodes each
batch once, instead of decoding every row and adding the rows with a float scatter-add. INT8
trains 1.19× (1M × 128) and 1.56× (4M × 128) faster than v0.2.6 and as fast as or faster than
v0.2.5 there (1.01× and 1.26×; same-day runs of each version); at 200k × 64 it is about as fast
as v0.2.6 and at 0.71× of v0.2.5, where BF16 is at 0.78×. Runs now repeat bit for bit (see Known limitations). FP8 is
unchanged: 0.65× to 0.81× of v0.2.5.

### What the RDNA4 tuning gains

Assignment kernel, tuned tables against the NVIDIA-fitted settings (`tools/tune_assign.py
--compare`, d = 32 to 768 (FP8: 64 to 768), k = 1,024 to 16,384, re-measured 2026-11-20). INT8
results are identical; for the other formats the tool accepts a configuration when the labels
are the same or, where tie order differs, the distances agree within rtol 1e-5 / atol 1e-4.

| Format | 65,536-row batches | 2,048-row batches |
|---|---|---|
| BF16 / FP16 | 1.06× to 2.80× | 0.95× to 3.45× |
| Native FP32 | 1.16× to 1.40× | 1.04× to 1.34× |
| INT8 | 2.99× to 8.51× | 2.57× to 10.4× |
| FP8 | 1.22× to 2.74× | 0.99× to 2.75× |

Training, each format with its tables on against off (3 warm epochs, v0.2.6; INT8 v0.2.7;
`tools/bench_train.py --no-tables` gives the "off" side, all AMD tables cleared, including the
second-output rules below):

| Format | 200k × 64 | 1M × 128 | 4M × 128 |
|---|---|---|---|
| BF16 | 0.97× | 1.42× | 2.05× |
| INT8 | 1.01× | 1.42× | 2.28× |
| FP8 | 0.99× | 1.43× | 1.89× |

The split-K choice (below) does not affect training, whose batches are large.

### Findings

- **Keep BF16 on RX cards.** RDNA has no FP32 matrix instruction, so native FP32 costs 1.2× to
  6.9× the time of BF16 here, more as the problem grows. Triton is still the faster FP32 path,
  so it stays the automatic choice. Since v0.2.6 BF16 is also faster than FP8, and than INT8
  up to 1M rows (next finding but one).
- **The NVIDIA-fitted tile settings are slow on RDNA4** (wave32): mostly from 128-row tiles at
  d > 128, and the stage count and tile shape on the full-row path (64 < d ≤ 128).
- **Cost-based revival changes the INT8/FP8 balance (v0.2.6).** Revival needs every training row's
  second-nearest distance and its decoded value. With the second-nearest output, the FP8 and
  INT8 kernels ran 1.4× to 2.3× slower under the v0.2 tables (d = 64 and 128, k = 4,096). Two
  rules for that output win at every batch size they cover (timed from 2,048 to 262,144 rows,
  k = 1,024 and 16,384): FP8 for 64 < d ≤ 128 from 8,192 rows (1.46× to 2.31×, measured at
  d = 128) and INT8 for d ≤ 64 from 16,384 rows (1.15× to 1.74×, measured at d = 64); the other
  candidates won only on large batches (`tools/tune_assign.py --second`). Upstream computed the encoded rows' revival terms
  with PyTorch scatter-adds (110 ms of GPU time in a 3-epoch INT8 run at 1M × 128, plus the
  decoding); the fused kernels, which now also decode the codes, take 7.5 ms (INT8) and 8.9 ms
  (FP8), of which 5.7 ms and 5.3 ms are the EMA update of the axes (`_removal_update_kernel`).
- **INT8 and FP8 need their own settings even more** (v0.2 measurements, before cost-based
  revival). With the NVIDIA-fitted settings for both,
  INT8 trained 0.83× to 1.27× as fast as BF16; with the tuned ones for both, 1.00× to 1.37×
  (same run as the tuning-gain table). The largest INT8 kernel gains
  (5× to 10×) are at d ≤ 64, where upstream pads every row to 128. One pipeline stage won on
  every INT8 and FP8 shape swept.
- **Settings tuned at one batch size can lose at another.** The v0.1 BF16 settings for d ≤ 64,
  tuned at 65,536 rows, ranged from 0.55× to 1.72× of upstream's speed between 2,048 and
  262,144 rows (128-row tiles at d ≤ 32 leave work-groups idle on small batches). One
  configuration chosen over that whole range now stays at 0.96× to 1.69×.
- **Split-K pays only on very small row counts on RDNA4.** At 256 rows it is 1.1× to 3.9× faster
  in all but one case (d = 64 with 4,096 centroids: 0.71×, still chosen). At 1,024 and 4,096 rows
  it is 1.6× to 3.8× slower, yet the NVIDIA-fitted rule chose it up to 4,032 rows at d = 64 and
  8,128 rows at d ≥ 128, so RDNA4 now uses it up to 512 rows only (`tools/tune_split_k.py`). The
  cap passes up a 1.02× to 1.14× gain at 1,024 rows with d = 128.
- **FP8 encodes exactly like PyTorch.** Every FP32 value up to 448 in magnitude (2.28 billion)
  gives the same E4M3 code through Triton's conversion as through
  `tensor.to(torch.float8_e4m3fn)` (an opt-in test, `EMEANS_EXHAUSTIVE_TESTS=1`; the default run
  samples), and so do the scaled and affine encodes through a quantizer.
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
| Kernels | The 10 PTX inline-assembly snippets in the EMA-update, split and FP8-cast kernels are replaced by plain Triton arithmetic. With fusion off, on NVIDIA (sm90) every float multiply, add and subtract compiles to an explicitly rounded `.rn` instruction as the assembly did, with an FMA only in the affine FP8 cast, where the assembly had one; halving now multiplies by 0.5 instead of dividing by 2 (the same value for every input). On the four AMD families it compiles without FMA contraction. |
| Device detection | New `device_profile.py`: tells NVIDIA from AMD (ROCm reports both as `"cuda"`) and maps AMD chips to a family with its wavefront size, LDS budget and matrix-instruction support. |
| Backend selection | `assignment_backend="auto"` picks Triton for FP32, FP16 and BF16 on the four families, so the default BF16 path works. Without Triton, auto falls back to PyTorch there (FP16/BF16 through `torch.addmm(out_dtype=)`, as explicit `native` already could upstream). The default FP32 → BF16 conversion is limited to the four families; other AMD GPUs keep FP32. CuTe and TMA descriptors are never selected on AMD. |
| INT8 (v0.2) | INT8/UINT8 assignment on the four families' integer matrix instructions, with a per-family pipeline depth that fits in 64 KiB of LDS (upstream's three stages need up to 112 KiB outside gfx942). |
| FP8 (v0.2) | FP8 (E4M3) assignment on the families with a native E4M3 matrix instruction, gfx950 and gfx120x. The scaled split's NVIDIA-only `libdevice.ldexp` is replaced on AMD by an exact power-of-two multiply (bit-identical to `torch.ldexp`). gfx942 and gfx11xx refuse FP8 with an error. |
| Upstream sync (v0.2.6) | faiss `b14b3d69` brought new PTX (revival statistics, the FP64 split step), now plain Triton arithmetic with fusion off (bit-equal to PyTorch on gfx1200); 12 new kernels checked for LDS (`tools/rocm_probe/lds_sweep.py`); an NVIDIA "SM90" batch-sum rule that also matched CDNA no longer does; NumPy fallbacks for three faiss C++ helpers missing from faiss-cpu 1.16.0; INT8/FP8 revival statistics in the fused Triton kernel, codes decoded there, instead of PyTorch scatter-adds. |
| Tuning | Per-family settings in `triton_assignment.py`, tuned on RDNA4: `_AMD_DIRECT_CONFIGS` (FP32, FP16/BF16, FP8), `_AMD_INT8_CONFIGS` with `_AMD_INT8_SMALL_CONFIGS` (batches up to 4,096 rows; 2,048 at d = 256), `_AMD_SPLIT_K_MAX_ROW_BLOCKS`, and since v0.2.6 `_AMD_DIRECT_SECOND_CONFIGS` / `_AMD_INT8_SECOND_CONFIGS` for launches that also return the second-nearest distance. gfx950 gets a cap to fit 64 KiB of LDS (`_AMD_DIRECT_CONFIGS["cdna4"]`, `_AMD_SPLIT_K_STAGES`). Other families keep the NVIDIA-fitted settings; `_direct_config` is the one place the launcher's choice is made. Gains are under Results. |
| Pre-flight checks (v0.2.2) | New `capabilities.py`: `supported(device, dtype, ...)` names the backend training would use, or `None` (`explain=True` adds the reason; v0.2.5: answers without torch installed); `memory_estimate(model, n, dtype, device)` bounds the GPU memory of `train`. New `check_finite=True` on `Emeans`/`QuantizedEmeans` raises on NaN or infinite input. Upstream has none of these. |
| Graph replay | Training uses the eager loop on ROCm (same results, only speed differs); `EMEANS_ROCM_GRAPH_REPLAY=1` opts in to replay. |
| Tests | Vendor-aware skips replace the ROCm skips; INT8 and FP8 tests run on AMD. New GPU-free tests: no inline assembly and `.rn` arithmetic (four AMD targets plus sm90), NVIDIA-only features fenced off, LDS fit of every launch configuration on four AMD targets, pinned launch configurations (NVIDIA unchanged), rule boundaries. New GPU tests: bit-level checks against PyTorch, the FP8 encodes included. One upstream IVF test made robust to OpenBLAS builds. |

Supported data per GPU and the converters are in the kernel table of the
[e-means README](https://github.com/ogaude/e-means-rocm/blob/main/contrib/e_means/README.md#kernels).

## Known limitations

- **Only RX 9000 (gfx120x) has been run.** MI300 (gfx942), MI350 (gfx950) and RX 7000 (gfx11xx)
  compile and are selected automatically, but are not validated and not tuned. MI300X is
  planned for v0.3.
- **gfx950 (MI350) is capped to fit 64 KiB of LDS, to be tested on hardware.** Its 128-row
  BF16/FP16 tile at d > 128 and two-stage split-K compile to 96 KiB on gfx950, more than the
  64 KiB the device profile assumes until MI350's limit is measured. So gfx950 uses 64-row tiles
  and one split-K stage. **To test on an MI350:** read its LDS limit (`tools/device_facts.py`),
  and if it is larger, raise `lds_bytes` for cdna4, drop the cap and time both.
- **The 64 KiB LDS fit is checked with AMD's ROCm Triton build only.** Compiled with upstream
  Triton from PyPI (3.8.0 and 3.9.0, on 2026-11-23), several launch configurations need 72 or
  96 KiB: gfx950 FP32 at every d, gfx1100 BF16 at d >= 192, and the BF16 split-K kernel on
  gfx1100 and gfx1200. AMD's build (`3.8.0+git669b31ac.rocm10.1.0`) fits them all. A ROCm torch
  that brings an upstream-based Triton (such as PyTorch's own ROCm wheels) is untested and may
  fail to launch these kernels.
- **FP8 runs only where E4M3 is native** (gfx120x, gfx950). MI300 and RX 7000 refuse it;
  emulated FP8 there is a possible v0.4 item.
- **NVIDIA has not been re-run on hardware.** "NVIDIA results unchanged" rests on mocked-device
  tests (backend selection, launch configurations) and on sm90 compiles of the rewritten
  kernels; the bit-level numerics tests have only run on AMD.
- **Several GPUs** (`DistributedEmeans`, sharded centroids) are expected to work on AMD but are
  untested until v0.5; their tests stay skipped on ROCm.
- **Training is not always bit-reproducible by default**, on AMD as upstream on NVIDIA. Since the
  upstream b14b3d69 sync, with the Triton backend and cluster revival on (the default, cost-based or
  mass-based), floating-point batch sums are added from label-sorted rows without atomics, up to
  d = 4,096 and batches of 2^24 rows; with revival off (`unused_threshold=0`), FP32/BF16 sums at d ≥ 128 use a segmented
  reduction without atomics. Other paths (revival off at d < 128, d > 4,096; encoded rows on the PyTorch or
  CuTe backend; FP8 sums in code units, repeatable in the runs below) may sum with atomic additions, so repeated runs can end in different clusterings.
  Measured on the RX 9060 XT (d = 768, k = 1,024, Triton, revival on, default settings): FP32,
  FP16, BF16 and FP8 runs are bit-identical without deterministic mode (v0.2.5: only FP16 and
  FP8). INT8 runs differed in v0.2.6; since v0.2.7 a stock `Int8Quantizer` sums its integer
  codes exactly and decodes once per batch, so INT8 repeats bit for bit too. Revival off,
  d < 128 and d > 4,096 are not measured.
  `torch.use_deterministic_algorithms(True)` makes runs identical on every path measured,
  including INT8.

## Development

Recorded development stack: ROCm 10.1.0, Python 3.14, torch 2.14.0+rocm10.1.0,
triton 3.8.0+git669b31ac.rocm10.1.0, faiss-cpu 1.16.0, on an RX 9060 XT (gfx1200, 32 CUs, wave32, 64 KiB
LDS per workgroup, 16 GB).

### Running the tests

The tests need `python -m pip install pytest`.

```bash
cd tests
# Everything except the GPU tests, without touching the GPU:
ROCR_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -m pytest -q $(ls *e_means*.py | grep -v -e utils -e conditions)
# Full run on a GPU (add EMEANS_ROCM_GRAPH_REPLAY=1 to include the graph-replay tests):
python -m pytest -q $(ls *e_means*.py | grep -v -e utils -e conditions)
# The same tests against the installed emeans_rocm package (no symlink needed):
PYTHONPATH=../tools python -m pytest -q -p pkg_alias $(ls *e_means*.py | grep -v -e utils -e conditions)
```

The first two commands need the development setup (see Quick start). The third uses the
`tools/pkg_alias.py` plugin, which maps `faiss.contrib.e_means` imports onto `emeans_rocm`; it
skips the three tests that check the faiss layout itself.

- Hide the GPU with `ROCR_VISIBLE_DEVICES=` (empty). `HIP_VISIBLE_DEVICES=-1` makes ROCm 10.1
  abort at the first `torch.cuda.is_available()`.
- GPU-free checks worth knowing: `torch_test_e_means_ptx_free.py` (no inline assembly, no FMA
  contraction on four AMD targets plus sm90), `torch_test_e_means_amd_fences.py` (NVIDIA-only
  features stay off on AMD; RDNA4 table shape and LDS fit), `torch_test_e_means_int8_compile.py`
  (INT8 LDS fit on four AMD targets), `torch_test_e_means_launch_config.py` (LDS fit of every
  launch configuration on four AMD targets, pinned configurations, rule boundaries),
  `torch_test_e_means_backend_selection.py` and `torch_test_e_means_device_profile.py` (mocked
  devices).
- On the GPU, `torch_test_e_means_triton_numerics.py` checks the rewritten kernels bit for bit
  against PyTorch and correctly rounded references, and `torch_test_e_means_fp8_amd.py` checks
  the FP8 encodes (plain, scaled, affine) and the `ldexp` replacement against PyTorch;
  `EMEANS_EXHAUSTIVE_TESTS=1` adds the encode of every FP32 value up to 448.
- `torch_test_e_means_ivf.py::TestIVFFlat::test_all_probes_split_tiles_empty_lists_and_ties`
  failed by one rounding step on this torch build, also on the unmodified upstream snapshot:
  the ROCm wheels use OpenBLAS, whose FP32 CPU matmul results depend on the matrix shape, so the
  index's batched products and the test's single product could differ in the last bit. The
  test now requires identical ids and scores within FP32 rounding of the represented operands;
  on the test's data, unrepresented (FP32) operands miss that bound by about 100 times.

### Tools

```bash
# Smoke matrix: every precision x backend, each case in its own process with a timeout
python tools/w0_smoke.py --rows 20000 --k 256          # small, safe on a shared GPU
python tools/w0_smoke.py                               # 200k x 64, k = 1024
EMEANS_ROCM_GRAPH_REPLAY=1 python tools/w0_smoke.py    # same with graph replay

# Training benchmark (the Results tables); add --no-tables for the NVIDIA-fitted settings
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
tuning sequence on a fresh cloud GPU; see their READMEs. `python tools/check_dist.py dist/`
checks a built wheel and sdist before release (package files, nothing local, version).

### Continuous integration

GitHub Actions has no GPU, so CI runs the GPU-free part of the suite; GPU runs stay local.

- `ci.yml`, on every push to `main` and every pull request: the GPU-free tests with a CPU-only
  torch on Python 3.10 to 3.14 (Triton compile checks skip), and the package job: build,
  `tools/check_dist.py`, `twine check`, the package without torch, and the tests against the
  installed package.
  A lint job runs ruff's pyflakes rules (undefined names, redefinitions; unused imports and
  variables only in `tools/`, since the other folders mirror faiss). Locally: `ruff check .`.
- `ci-rocm-stack.yml`, weekly and on demand: the GPU-free tests on AMD's ROCm torch and Triton
  (about 6.6 GB installed), which adds the Triton compile checks for the four AMD targets.

## Roadmap

The work runs on two tracks. The **GPU track** adds and tunes hardware; each of its versions
needs the hardware it names. The **integration track** ships what callers of the library need,
as patch releases of the current version, and needs no new hardware. Since v0.2.1 only the
integration track has moved: v0.3 is waiting for MI300X access.

**GPU track**

| Version | Scope | Hardware | Status |
|---|---|---|---|
| **v0.1** | Triton as the default AMD path for FP32, FP16 and BF16; validated and tuned on RDNA4 | RX 9060 XT | **released** 2026-11-19 |
| **v0.2** | INT8 on all four families; native FP8 on gfx120x and gfx950; RDNA4 tuning for both, small-batch settings and the split-K choice; the cloud session tooling | RX 9060 XT | **released** 2026-11-20; v0.2.1 (review fixes) the same day |
| v0.3 | MI300X (gfx942) validation and tuning; measure whether a hand-written AMD kernel (the CuTe counterpart) is worth building | MI300X (cloud) | planned; scripts ready; **waiting for cloud access** |
| v0.4 | Optionally emulated FP8 on gfx942 and gfx11xx; the hand-written kernel if v0.3's measurement calls for it | RX 9060 XT and MI300X | planned |
| v0.5 | Several GPUs (`DistributedEmeans`, sharded centroids) | 8× MI300X (cloud) | planned |

The MI300X work was first planned for v0.2; it moved to v0.3 (2026-11-20) so the RX 9000 work
could be released without waiting for cloud access, and the later versions moved up by one.

**Integration track**

These came from a downstream integration of e-means, which had to work around missing checks
and the way the fork was installed.

| Release | Scope | Status |
|---|---|---|
| **v0.2.2** | Pre-flight checks: `supported()`, `memory_estimate()`, `check_finite=True` | **released** 2026-11-22 |
| **v0.2.3** | `Emeans` refuses INT8/UINT8 rows on the GPU with an error naming `QuantizedEmeans` | **released** 2026-11-22 |
| **v0.2.4** | Installable package `emeans-rocm` (import name `emeans_rocm`) on PyPI; users no longer need the symlink into faiss-cpu (development still uses it) | **released** 2026-11-23 |
| **v0.2.5** | `supported()` and `memory_estimate()` answer without torch installed; `supported(..., explain=True)` gives the reason; a smaller wheel | **released** 2026-11-23 |
| **v0.2.6** | Synced with upstream faiss `b14b3d69` (cost-based revival by default, fractional epochs, BF16 IVF training and two-nearest IVF search); ROCm work for the new code; INT8/FP8 revival speed; CPU path on faiss-cpu 1.16.0 | **released** 2026-11-26 |
| **v0.2.7** | INT8 training repeats bit for bit and is 1.19× to 1.56× faster at 1M to 4M rows (exact integer code sums); determinism measured on AMD | **released** 2026-11-27 |

Not scheduled on either track:

- **Upstream.** Whether faiss would take the ROCm changes into `contrib/e_means` has not been
  asked yet. The fork keeps faiss's layout so its changes still apply there
  ([Relation to upstream faiss](https://github.com/ogaude/e-means-rocm#relation-to-upstream-faiss)); if they were merged, the package
  could shrink or be retired.
- **NVIDIA hardware run.** NVIDIA paths rest on mocked selection tests and sm90 compiles (see
  Known limitations); a run needs access to an NVIDIA GPU.

**v0.3 plan (needs an MI300X)**

| Item | Plan |
|---|---|
| gfx942 validation | smoke matrix, test suite (eager and replay), INT8 there |
| gfx942 tuning | `cdna3` entries for the BF16, INT8 and split-K tables; graph-replay default decided on measured speed |
| Hand-written backend gate | Triton BF16 assignment against rocBLAS/hipBLASLt GEMM (`tools/gemm_gate.py`); build a FlyDSL backend (v0.4) only if Triton reaches less than about 80% of the GEMM's speed |
| MI350 (gfx950), if the cloud offers one | measure its LDS limit and lift or keep the 64 KiB cap (see Known limitations) |

### Release history

**v0.2.7 (2026-11-27, tag `e-means-rocm-v0.2.7`)**: INT8 batch sums are exact. A stock
`Int8Quantizer` sums its integer codes per centroid (|code| ≤ 128, so FP32 partial sums stay
exact integers up to 131,072 rows per batch, whatever the addition order) and decodes each batch
once as `sum · 2^-e + count · shift`; before, every row was decoded and the rows were added with
a float scatter-add, whose rounding depended on the order of the atomic additions. Other
encoded rows on the Triton backend (batches above 131,072 rows up to 2^24, other quantizers,
d ≤ 4,096) are added from label-sorted rows, without atomics, outside deterministic mode. Two
INT8 runs with the same seed now give bit-identical centroids without deterministic mode; INT8
results change in the last bits compared with v0.2.6 (final objective within 0.03% of v0.2.6's runs, which differ among
themselves by up to 0.02%). INT8 trains 1.19× and 1.56× faster than v0.2.6
at 1M × 128 and 4M × 128 (about as fast at 200k × 64); FP32, FP16, BF16 and FP8 results are
unchanged bit for bit on the default paths.
Determinism measured on the RX 9060 XT (see Known limitations): FP32, FP16, BF16, FP8 and now
INT8 Triton runs repeat bit for bit with revival on (v0.2.5: only FP16 and FP8). Tests: 387
passed and 88 skipped on gfx1200 (391 passed with graph replay); two new INT8 tests.

**v0.2.6 (2026-11-26, tag `e-means-rocm-v0.2.6`)**: synced with upstream faiss `b14b3d69`
(2026-11-25). From upstream: cost-based cluster revival is the default (`unused_threshold=None`;
a number keeps mass-based revival, `0` turns revival off), so the same seed and data give
different centroids than v0.2.5; fractional epochs (`nepochs=1.6`), whose last pass also
reassigns the rows of clusters split during it, so the returned labels match the final centroids
(see Cluster revival in the [e-means README](https://github.com/ogaude/e-means-rocm/blob/main/contrib/e_means/README.md));
BF16 IVF training and a two-nearest IVF search. For ROCm: the new upstream PTX (revival statistics, the FP64 split
step) is plain Triton arithmetic, bit-equal to PyTorch on gfx1200; the 12 new kernels fit 64 KiB
of LDS on the four families; an NVIDIA "SM90" rule no longer matches CDNA; `NumpyEmeans` runs on
faiss-cpu 1.16.0, which lacks three C++ helpers upstream now calls (NumPy fallbacks). Speed on an
RX 9060 XT: BF16/FP16 train 1.3× to 1.9× faster than in v0.2.5 at 1M and 4M rows (0.88× and
0.95× as fast at 200k × 64); INT8 and FP8 train at 0.58× to 0.85× of BF16's speed, because
revival needs each row's second-nearest distance and decoded value. Right after the sync they
were 2.2× to 2.6× slower than in v0.2.5 at 1M × 128; fused revival statistics with in-kernel
decoding and two launch rules for the second-nearest output bring that to 1.2× there (1.2× to
1.6× across the three shapes; see Results). Tests: 385 passed and 88
skipped on gfx1200 (graph replay also passes); GitHub CI passes, including the ROCm-stack run.

**v0.2.5 (2026-11-23, tag `e-means-rocm-v0.2.5`)**: `supported()` and `memory_estimate()` answer
without PyTorch installed, for example on a CPU-only machine that only uses `NumpyEmeans`:
`supported` returns `None`, and `memory_estimate` returns 0 for the CPU and raises a
`RuntimeError` naming the missing torch for a GPU. Both take dtype names (`"bfloat16"`) and
device strings. `supported(..., explain=True)` returns `(backend, reason)`, for example
`(None, "torch is not installed")`. When torch has no GPU, asking about a GPU device raised
instead of answering: on a CPU-only torch build for every backend ("Torch not compiled with CUDA
enabled", torch 2.15 CPU), and on ROCm torch with the GPU hidden for an explicit `native` or
`triton` backend with BF16 ("No CUDA GPUs are available"). It now returns `None` with torch's
reason, and `memory_estimate` raises a `RuntimeError` saying torch sees no GPU. The wheel leaves out the README figures (734 KB to 160 KB),
and `pip install -e .` now stops with an error instead of installing a copy. No other change to
the e-means code. 291 tests passed eager, 295 with graph replay, 232 without a GPU, none failing;
against the package (eager), 288 passed and the 3 that check the faiss layout were skipped. The
capabilities tests also passed on a CPU-only torch 2.15 build.

**v0.2.4 (2026-11-23, tag `e-means-rocm-v0.2.4`)**: the fork installs as a package,
`pip install emeans-rocm`, imported as `emeans_rocm`. Before, using it meant replacing the
`faiss/contrib/e_means` folder inside an installed faiss-cpu, and a reinstall of faiss-cpu
through that symlink could delete files in the clone; the development setup still works that way. `pyproject.toml` builds the wheel straight
from `contrib/e_means`, so the source keeps faiss's layout; `tools/pkg_alias.py` runs the
unchanged tests against the installed package. No change to the e-means code. 285 tests passed
eager, 289 with graph replay, none failing; against the package (eager), 282 passed and the 3
that check the faiss layout were skipped, besides the usual hardware skips.

**v0.2.3 (2026-11-22, tag `e-means-rocm-v0.2.3`)**: `Emeans` refuses INT8/UINT8 rows on the GPU at
the first `train`, `accumulate` or `assign` call, naming `QuantizedEmeans`, instead of failing
inside the Triton launcher with a dtype-mismatch error. `QuantizedEmeans` and the CPU path are
unchanged. 285 tests passed eager, 289 with graph replay, none failing.

**v0.2.2 (2026-11-22, tag `e-means-rocm-v0.2.2`)**: checks a caller can run before training,
asked for by an integration that had to probe with a tiny training run. `supported(device,
dtype, ...)` names the backend training would use, or `None` when it would be refused, from
device facts and installed software (it matched real training in all 96 format × mode ×
backend cases on gfx1200 and the CPU). `memory_estimate(model, n, dtype, device)` bounds the GPU
memory of `train` (1.07× to 1.33× the measured peak on gfx1200, never below). `check_finite=True`
on `Emeans`/`QuantizedEmeans` raises on NaN or infinite input. New test that deterministic mode
gives bit-identical training runs. 284 tests passed eager, 288 with graph replay, none failing.

**v0.2.1 (2026-11-20, tag `e-means-rocm-v0.2.1`)**: fixes from an independent review. gfx950
(MI350) capped to fit 64 KiB of LDS (its BF16/FP16 128-row tile and two-stage split-K needed
96 KiB; to be tested on hardware); a GPU-free test of every launch configuration on all four
AMD families; sm90 compiles of the split and FP8-cast kernels; FP8 quantized-encode parity and
an opt-in exhaustive encode test; a wrong test skip condition fixed; `bench_train.py
--no-tables`; documentation matched to code and tests. 269 tests passed eager, 273 with graph
replay, none failing.

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
pyproject.toml        builds contrib/e_means as the package emeans-rocm (import name emeans_rocm)
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
tools/pkg_alias.py    pytest plugin: run tests/ against the installed emeans_rocm package
tools/check_dist.py   checks a built wheel and sdist before release
.github/workflows/    CI: GPU-free tests and package checks (see Continuous integration)
tools/gpu_session/    setup + one-shot session runner for a cloud GPU VM (see its README)
```

## Relation to upstream faiss

This repository starts from [facebookresearch/faiss](https://github.com/facebookresearch/faiss)
`main` at commit `14392c952788ca1d6980361cf01d1c8736afc4db` (2026-11-18) and is synced to
`b14b3d693161c7b1cbc5f2e398c8234b42b04c8b` (2026-11-25). Upstream code enters only through
unmodified import commits: `d6e4215` (the first snapshot of `contrib/e_means`, `benchs/e_means`
and the e-means tests) and `7268fac` (the same paths at `b14b3d69`), each merged into `main`. So
`git diff 7268fac -- contrib benchs tests` shows exactly what this fork changes in faiss's
files. Those paths mirror faiss, so changes to them apply to a faiss checkout at `b14b3d69` (`git diff 7268fac main -- contrib benchs tests > fork.patch`, then `git apply fork.patch`); `README.md`, `pyproject.toml`,
`tools/` and `.gitignore` belong to this fork only. To see what upstream changed since the last sync:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/facebookresearch/faiss.git /tmp/faiss
git -C /tmp/faiss sparse-checkout set contrib/e_means benchs/e_means tests
mkdir -p /tmp/snapshot && git archive 7268fac contrib/e_means | tar -x -C /tmp/snapshot
diff -r /tmp/snapshot/contrib/e_means /tmp/faiss/contrib/e_means
```

## License

MIT, as faiss; see [LICENSE](https://github.com/ogaude/e-means-rocm/blob/main/LICENSE). The e-means code is by Meta Platforms, Inc. and
affiliates; the ROCm changes in this repository are under the same license.
