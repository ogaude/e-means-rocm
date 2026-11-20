#!/usr/bin/env bash
# Run the e-means GPU validation and tuning session and collect every result in one
# directory plus a tarball. Built for a short-lived cloud VM (MI300X), also works on a
# local card. Run from the repository root, inside the venv from setup.sh, under tmux
# so a dropped SSH connection does not stop it:
#
#     tmux new -s em
#     source ~/venvs/emeans-rocm/bin/activate
#     bash tools/gpu_session/run.sh                      # all steps, in the order below
#     STEPS="facts gate" bash tools/gpu_session/run.sh   # selected steps only
#     QUICK=1 bash tools/gpu_session/run.sh              # tiny shapes: dry run of the script
#
# Steps (default order: decision-critical and cheap first, the long sweep last):
#   facts         device facts JSON, rocminfo/rocm-smi snapshots, versions
#   probe         offline compile probe of the assignment kernels on this Triton (LDS, MFMA)
#   smoke         tools/w0_smoke.py, eager
#   smoke-replay  tools/w0_smoke.py, EMEANS_ROCM_GRAPH_REPLAY=1
#   gate          tools/gemm_gate.py: Triton BF16 assignment vs BLAS GEMM (C gate)
#   bench         tools/bench_train.py: warm train(), Triton vs native, eager vs replay
#   tests         pytest, every e-means test file, eager
#   tests-replay  same with EMEANS_ROCM_GRAPH_REPLAY=1
#   split-k       tools/tune_split_k.py: direct vs split-K on small row grids
#   tune-compare  tools/tune_assign.py --compare (upstream constants vs this GPU's table)
#   tune-sweep    tools/tune_assign.py sweep (longest step; hours)
#
# Each step logs to $OUT/<step>.log; a failing step is recorded and the next one runs.
# $OUT/steps.tsv lists step, exit code and minutes. Variables:
#   OUT           result directory (default results/<host>-<arch>-<UTC time>)
#   STEPS         space-separated step list (default: all, in the order above)
#   PYTHON        interpreter (default: python on PATH, i.e. the activated venv)
#   QUICK=1       small shapes everywhere (minutes; checks the script, not the GPU)
#   SWEEP_ARGS    extra arguments for the sweep (default: --quick --k 1024 16384, the
#                 grid behind the rdna4 table)
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
PY=${PYTHON:-python}
python() { "$PY" "$@"; }
ALL="facts probe smoke smoke-replay gate bench tests tests-replay split-k tune-compare tune-sweep"
STEPS=${STEPS:-$ALL}
QUICK=${QUICK:-0}
ARCH=$(python -c 'import torch; p=torch.cuda.get_device_properties(0); print(getattr(p,"gcnArchName","nvidia").split(":")[0])' 2>/dev/null || echo unknown)
OUT=${OUT:-$REPO/results/$(hostname -s)-$ARCH-$(date -u +%Y%m%dT%H%MZ)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
SWEEP_ARGS=${SWEEP_ARGS:---quick --k 1024 16384}

if [ "$QUICK" = 1 ]; then
  SMOKE_ARGS="--rows 20000 --k 256"
  GATE_ARGS="--rows 8192 --d 64 128 --k 1024 4096"
  BENCH_ARGS="--shapes 50000x64x256 --epochs 1 --repeat 1"
  SPLITK_ARGS="--rows 256 4096 --d 128 --k 4096"
  COMPARE_ARGS="--rows 8192 --d 64 256 --k 1024"
  SWEEP_ARGS="--quick --rows 8192 --dtypes bf16 --d 64 --k 1024 --top 3"
  TEST_FILES="torch_test_e_means_device_profile.py torch_test_e_means_triton_numerics.py"
else
  SMOKE_ARGS=""
  GATE_ARGS=""
  BENCH_ARGS=""
  SPLITK_ARGS=""
  COMPARE_ARGS=""
  TEST_FILES=$(cd tests && ls *e_means*.py | grep -v -e utils -e conditions | tr '\n' ' ')
fi

printf 'step\texit\tminutes\n' > "$OUT/steps.tsv"

run_step() {  # name, command...
  local name=$1; shift
  local log="$OUT/$name.log" t0 rc
  t0=$(date +%s)
  printf '\n===== %s  (%s)  -> %s\n' "$name" "$(date -u +%H:%M:%SZ)" "$log"
  {
    echo "# $*"
    echo "# started $(date -u +%FT%TZ)  EMEANS_ROCM_GRAPH_REPLAY=${EMEANS_ROCM_GRAPH_REPLAY:-unset}"
  } > "$log"
  "$@" 2>&1 | tee -a "$log"
  rc=${PIPESTATUS[0]}
  local minutes=$(( ($(date +%s) - t0 + 30) / 60 ))
  printf '%s\t%s\t%s\n' "$name" "$rc" "$minutes" >> "$OUT/steps.tsv"
  printf '===== %s: exit %s, %s min\n' "$name" "$rc" "$minutes"
}

step_facts() {
  git -C "$REPO" log -1 --format='%H %s' > "$OUT/git_head.txt"
  git -C "$REPO" status --short >> "$OUT/git_head.txt"
  python -m pip freeze > "$OUT/pip_freeze.txt" 2>&1
  { command -v rocminfo >/dev/null && rocminfo; } > "$OUT/rocminfo.txt" 2>&1
  { command -v rocm-smi >/dev/null && rocm-smi --showallinfo; } > "$OUT/rocm_smi.txt" 2>&1
  { command -v amd-smi >/dev/null && amd-smi static; } > "$OUT/amd_smi_static.txt" 2>&1
  { [ -r /opt/rocm/.info/version ] && cat /opt/rocm/.info/version; uname -a; nproc;
    free -g; df -h "$REPO"; } > "$OUT/host.txt" 2>&1
  python tools/device_facts.py --out "$OUT/device_facts.json"
}

step_probe() {
  local dir="$OUT/probe" target
  target=$(python -c 'import triton; print(triton.runtime.driver.active.get_current_target().arch)')
  mkdir -p "$dir"
  cp tools/rocm_probe/*.py "$dir/"
  # Compile for this GPU's target plus gfx942 (MI300) for comparison; runs offline.
  (cd "$dir" && python assign.py "$REPO/contrib/e_means/triton_assignment.py" \
      $(printf '%s\n' "$target" gfx942 | sort -u))
}

step_smoke() {
  python tools/w0_smoke.py $SMOKE_ARGS --out "$OUT/smoke.jsonl"
}

step_smoke_replay() {
  EMEANS_ROCM_GRAPH_REPLAY=1 python tools/w0_smoke.py $SMOKE_ARGS --out "$OUT/smoke_replay.jsonl"
}

step_gate() {
  python tools/gemm_gate.py $GATE_ARGS --out "$OUT/gemm_gate.jsonl"
}

step_bench() {
  python tools/bench_train.py $BENCH_ARGS --out "$OUT/bench_train.jsonl"
}

step_tests() {
  (cd tests && python -m pytest -q -rfE $TEST_FILES --junitxml="$OUT/$1.xml")
}

step_split_k() {
  python tools/tune_split_k.py $SPLITK_ARGS
}

step_tune_compare() {
  python tools/tune_assign.py --compare $COMPARE_ARGS
}

step_tune_sweep() {
  python tools/tune_assign.py $SWEEP_ARGS
}

echo "repo:  $REPO ($(git -C "$REPO" rev-parse --short HEAD))"
echo "arch:  $ARCH"
echo "out:   $OUT"
echo "steps: $STEPS"
[ "$QUICK" = 1 ] && echo "QUICK=1: small shapes; numbers are not meaningful"
command -v rocm-smi >/dev/null && rocm-smi --showuse --showmemuse 2>/dev/null | grep -E 'GPU\[' || true

for s in $STEPS; do
  case $s in
    facts)        run_step facts step_facts ;;
    probe)        run_step probe step_probe ;;
    smoke)        run_step smoke step_smoke ;;
    smoke-replay) run_step smoke-replay step_smoke_replay ;;
    gate)         run_step gate step_gate ;;
    bench)        run_step bench step_bench ;;
    tests)        run_step tests step_tests tests ;;
    tests-replay) EMEANS_ROCM_GRAPH_REPLAY=1 run_step tests-replay step_tests tests-replay ;;
    split-k)      run_step split-k step_split_k ;;
    tune-compare) run_step tune-compare step_tune_compare ;;
    tune-sweep)   run_step tune-sweep step_tune_sweep ;;
    *) echo "unknown step: $s (known: $ALL)" >&2 ;;
  esac
  # Keep a current tarball after every step, so results survive an early stop.
  tar -C "$(dirname "$OUT")" -czf "$OUT.tar.gz" "$(basename "$OUT")"
done

printf '\n===== summary\n'
column -t "$OUT/steps.tsv" 2>/dev/null || cat "$OUT/steps.tsv"
echo
echo "Results: $OUT"
echo "Tarball: $OUT.tar.gz"
echo "Copy it off the machine before destroying it, from your own computer:"
echo "    scp <user>@<vm-address>:$OUT.tar.gz ."
