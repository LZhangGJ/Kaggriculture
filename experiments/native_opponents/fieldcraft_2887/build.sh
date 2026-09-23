#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniforge3/envs/torch-npu/bin/python}
CXX_BIN=${CXX_BIN:-g++}
EXT_SUFFIX=$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
FAST_SO=$(find "$ROOT/fast_kaggriculture/python/fast_kaggriculture" \
  -maxdepth 1 -name '_fast_kaggriculture*.so' -print -quit)
[[ -n "$FAST_SO" ]] || { echo "missing built fast_kaggriculture extension" >&2; exit 1; }
read -r -a PY_INCLUDES <<<"$($PYTHON_BIN -m pybind11 --includes)"
mkdir -p "$HERE/build"

"$CXX_BIN" -O3 -DNDEBUG -std=c++20 -fPIC -shared -pthread \
  "${PY_INCLUDES[@]}" -I"$ROOT/fast_kaggriculture/src" \
  "$HERE/fieldcraft_2887.cpp" "$HERE/bindings.cpp" \
  -Wl,--no-as-needed "$FAST_SO" \
  -Wl,-rpath,"$(dirname "$FAST_SO")" \
  -o "$HERE/build/fieldcraft_2887_native$EXT_SUFFIX"

echo "$HERE/build/fieldcraft_2887_native$EXT_SUFFIX"
