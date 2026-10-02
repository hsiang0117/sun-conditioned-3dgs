#!/usr/bin/env bash
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
python="$repo/.venv/bin/python"
test -x "$python" || { printf '%s\n' 'Create this repository .venv first.' >&2; exit 1; }

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-4}"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export PATH="$CUDA_HOME/bin:$PATH"
test -x "$CUDA_HOME/bin/nvcc" || { printf 'Missing nvcc: %s\n' "$CUDA_HOME/bin/nvcc" >&2; exit 1; }
command -v "$CC" >/dev/null
command -v "$CXX" >/dev/null

cd -- "$repo"
mkdir -p temporary-build
"$python" -I - "$repo" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

repo = Path(sys.argv[1])
venv = repo / '.venv'
assert Path(sys.prefix).resolve() == venv.resolve(), 'Wrong Python environment'
assert Path(torch.__file__).resolve().is_relative_to(venv.resolve()), 'External torch import'
assert torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8', 'Wrong torch build'
assert torch.cuda.is_available(), 'CUDA device is unavailable'
extensions = {}
for name in ('diff-gaussian-rasterization', 'simple-knn', 'fused-ssim'):
    path = 'submodules/' + name
    expected = subprocess.check_output(['git', 'ls-tree', 'HEAD', path], text=True).split()[2]
    actual = subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True).strip()
    assert expected == actual, f'Wrong extension commit: {path}'
    extensions[name] = actual
headers = Path(torch.__file__).parent / 'include'
header_hashes = {relative: hashlib.sha256((headers / relative).read_bytes()).hexdigest()
                 for relative in ('torch/csrc/dynamo/compiled_autograd.h',
                                  'c10/cuda/CUDACachingAllocator.h')}
state = {'python': sys.executable, 'python_version': sys.version,
         'torch': torch.__version__, 'torch_file': torch.__file__,
         'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
         'cuda_home': os.environ['CUDA_HOME'], 'cuda_arch': os.environ['TORCH_CUDA_ARCH_LIST'],
         'max_jobs': os.environ['MAX_JOBS'], 'cc': os.environ['CC'], 'cxx': os.environ['CXX'],
         'extension_commits': extensions, 'torch_header_sha256': header_hashes}
(repo / 'temporary-build/build-environment-linux.json').write_text(json.dumps(state, indent=2))
print(json.dumps(state, indent=2))
PY

"$CUDA_HOME/bin/nvcc" --version
"$CXX" --version | head -n 1
for extension in diff-gaussian-rasterization simple-knn fused-ssim; do
    "$python" -I -m pip install --no-build-isolation --no-deps --force-reinstall \
        "$repo/submodules/$extension" --log "$repo/temporary-build/$extension-linux.log"
done
"$python" -m unittest discover -s tests -p test_sun_conditioning.py -v
"$python" -I -m pip check
"$python" -I - <<'PY'
import torch
from fused_ssim import fused_ssim
x = torch.rand((1, 3, 64, 64), device='cuda', requires_grad=True)
y = torch.rand_like(x)
value = fused_ssim(x, y)
(1 - value).backward()
assert torch.isfinite(value) and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
print('fused-SSIM CUDA forward/backward: OK')
PY
