#!/usr/bin/env bash
# Prepare a fresh GPU machine (for example an AMD Developer Cloud MI300X VM) for
# tools/gpu_session/run.sh. Run from the repository root:
#
#     bash tools/gpu_session/setup.sh
#
# Creates a virtual environment (default ~/venvs/emeans-rocm), installs a ROCm build of
# PyTorch + Triton, faiss-cpu, numpy and pytest, and points faiss.contrib.e_means at
# this checkout (symlink; the wheel's copy is kept as e_means.pypi-orig).
#
# Environment overrides:
#   VENV          venv path                         (default ~/venvs/emeans-rocm)
#   PYTHON        interpreter for the venv          (default python3)
#   GFX           target for the torch wheel extra  (default: detected with rocminfo)
#   TORCH_INDEX   pip index for ROCm torch          (default AMD whl-next index)
#   TORCH_SPEC    pip requirement for torch         (default: same stack as v0.1,
#                 "torch[device-$GFX]==2.14.0+rocm10.1.0")
#   USE_SYSTEM_TORCH=1  no venv: use the ROCm torch already importable by $PYTHON (for
#                 example inside a ROCm PyTorch container, PYTHON=/opt/venv/bin/python)
#                 and install the other packages into that environment
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
VENV=${VENV:-$HOME/venvs/emeans-rocm}
PYTHON=${PYTHON:-python3}
TORCH_INDEX=${TORCH_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}

say() { printf '\n== %s\n' "$*"; }

say "GPU"
if command -v rocminfo >/dev/null; then
  rocminfo | grep -E '^\s+Name:\s+gfx' | sort | uniq -c || true
  GFX=${GFX:-$(rocminfo | grep -oE 'gfx[0-9a-f]+' | head -1)}
elif command -v amd-smi >/dev/null; then
  amd-smi static --asic 2>/dev/null | grep -iE 'target|market' | head -4 || true
fi
if [ -z "${GFX:-}" ]; then
  echo "No AMD GPU found by rocminfo; set GFX=gfx942 (or similar) to continue." >&2
  exit 1
fi
echo "target: $GFX"
command -v rocm-smi >/dev/null && rocm-smi --showuse --showmemuse 2>/dev/null | grep -E 'GPU\[' || true
[ -r /opt/rocm/.info/version ] && echo "host ROCm: $(cat /opt/rocm/.info/version)"
echo "kernel: $(uname -r)"

TORCH_SPEC=${TORCH_SPEC:-"torch[device-$GFX]==2.14.0+rocm10.1.0"}

if [ "${USE_SYSTEM_TORCH:-0}" = 1 ]; then
  say "using $PYTHON (no venv)"
  "$PYTHON" -c 'import torch; assert torch.version.hip, "torch is not a ROCm build"; print("torch", torch.__version__)'
  PY=$PYTHON
  ACTIVATE="PYTHON=$PYTHON"
else
  say "venv $VENV"
  "$PYTHON" -m venv "$VENV"
  PY=$VENV/bin/python
  ACTIVATE="source $VENV/bin/activate &&"
  "$PY" -m pip install -q --upgrade pip
  say "torch: $TORCH_SPEC from $TORCH_INDEX"
  "$PY" -m pip install --index-url "$TORCH_INDEX" "$TORCH_SPEC"
fi
"$PY" -m pip install -q faiss-cpu numpy pytest

say "point faiss.contrib.e_means at $REPO/contrib/e_means"
C=$("$PY" -c "import faiss.contrib, os; print(os.path.dirname(faiss.contrib.__file__))")
if [ -L "$C/e_means" ]; then
  ln -sfn "$REPO/contrib/e_means" "$C/e_means"
else
  [ -e "$C/e_means" ] && mv "$C/e_means" "$C/e_means.pypi-orig"
  ln -s "$REPO/contrib/e_means" "$C/e_means"
fi

say "check"
"$PY" - <<'EOF'
import torch, triton, faiss.contrib.e_means as em, os
print("torch", torch.__version__, "hip", torch.version.hip, "| triton", triton.__version__)
print("e_means from", os.path.realpath(os.path.dirname(em.__file__)))
print("cuda available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device 0:", p.name, getattr(p, "gcnArchName", "?"),
          "| CUs:", p.multi_processor_count, "| %.0f GiB" % (p.total_memory / 2**30))
    x = torch.randn(4096, 64, device="cuda")
    print("matmul ok:", float((x @ x.T).trace()) > 0)
EOF
echo
echo "Ready. Next (inside tmux): $ACTIVATE bash tools/gpu_session/run.sh"
