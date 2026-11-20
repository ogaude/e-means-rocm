# GPU session scripts

These scripts run the whole e-means validation and tuning sequence on one GPU machine
and collect the results. They are built for a short-lived cloud VM (the v0.2 MI300X
session), so the VM is created, used and destroyed in one sitting. They also run on a
local card.

| Script | Does |
|---|---|
| `setup.sh` | Creates the venv, installs ROCm PyTorch + Triton, faiss-cpu, numpy and pytest, and points `faiss.contrib.e_means` at this checkout |
| `run.sh` | Runs the session steps, logs each one, and keeps an up-to-date tarball of the results after every step |

The Python tools it calls also run on their own:

| Tool | Measures |
|---|---|
| `tools/device_facts.py` | Arch name, CU count, LDS, Triton target, e-means device profile, `addmm(out_dtype=)`, graph capture, BLAS libraries |
| `tools/w0_smoke.py` | Every precision × backend trains (or refuses with the intended error) |
| `tools/gemm_gate.py` | C gate: Triton BF16 assignment time against a bare BLAS GEMM of the same shape |
| `tools/bench_train.py` | Warm `train()`: Triton vs PyTorch, eager vs graph replay |
| `tools/tune_split_k.py` | Direct kernel vs split-K at each partition count, on small row grids |
| `tools/tune_assign.py` | Tile/warp/stage sweep (`--compare`: upstream constants vs this GPU's table) |

## Before booking the VM

1. Do a dry run on the local card: `QUICK=1 bash tools/gpu_session/run.sh` (a few minutes;
   uses tiny shapes, so the numbers mean nothing).
2. Push everything the session should test. The VM clones the public repository.
3. Note the timings below. On the AMD Developer Cloud a **stopped VM keeps billing until it is
   destroyed**, and the VM and its disk are deleted when the credit runs out.

## On the VM

```bash
git clone https://github.com/ogaude/e-means-rocm && cd e-means-rocm
bash tools/gpu_session/setup.sh        # about 5 to 10 minutes (wheel download)
tmux new -s em                         # survives a dropped SSH connection
source ~/venvs/emeans-rocm/bin/activate
bash tools/gpu_session/run.sh 2>&1 | tee session.out
```

`setup.sh` pins the same stack as v0.1 (torch 2.14.0+rocm10.1.0, wheel extra
`device-gfx942` detected from `rocminfo`). The wheels bundle the ROCm runtime and only
need the kernel driver. To change the version, set `TORCH_SPEC`. If the VM image (or a
ROCm PyTorch container) already has a ROCm PyTorch you prefer, skip the venv with
`USE_SYSTEM_TORCH=1 PYTHON=/path/to/python bash tools/gpu_session/setup.sh`, then run the
session with the same `PYTHON=/path/to/python`.

## Session steps and the decisions they feed

Steps run in this order, cheap and decision-critical first. Select steps with
`STEPS="facts gate"`.

| Step | Output | Feeds |
|---|---|---|
| `facts` | `device_facts.json`, `rocminfo.txt`, `rocm_smi.txt`, `pip_freeze.txt` | the cdna3 profile (LDS, CU count, `multi_processor_count`) |
| `probe` | LDS and matrix instruction per kernel on this Triton | per-family stage limits (INT8) |
| `smoke`, `smoke-replay` | `smoke*.jsonl` | gfx942 works end to end, eager and replay |
| `gate` | `gemm_gate.jsonl` and a verdict | C gate: build the FlyDSL backend or not |
| `bench` | `bench_train.jsonl` and a summary table | Triton vs native on gfx942; graph-replay default for cdna3 |
| `tests`, `tests-replay` | pytest logs and JUnit XML | acceptance |
| `split-k` | direct vs split-K per shape | refit of `_SPLIT_K_TARGET_PROGRAMS` for 304 CUs |
| `tune-compare` | upstream vs table | baseline before a cdna3 entry exists |
| `tune-sweep` | best configurations per shape | the `cdna3` entry in `_AMD_DIRECT_CONFIGS` |

`tune-sweep` is the long one: about two hours on gfx1200 with the default grid
(`--quick --k 1024 16384`). Change it with `SWEEP_ARGS`.

## Second part, same sitting: check the new table

After the sweep, write the `cdna3` entry (and any split-K change) on your own machine, push
it, then on the VM:

```bash
git pull
STEPS="tune-compare bench tests gate" OUT=results/after-table bash tools/gpu_session/run.sh
```

## Getting the results off the VM

`run.sh` updates `results/<host>-<arch>-<time>.tar.gz` after every step and prints its path at
the end. From your own computer:

```bash
scp <user>@<vm-address>:e-means-rocm/results/<name>.tar.gz .
```

Destroy the VM only once the tarball is safe on your computer. `results/` is git-ignored.
