#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/root/miniforge3/envs/torch-npu/bin/python}
CXX_BIN=${CXX_BIN:-g++}
read -r -a OPT_FLAGS <<<"${OPT_FLAGS:--O3 -DNDEBUG -march=native}"
EXT_SUFFIX=$($PYTHON_BIN -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
FAST_SO=$(find "$ROOT/fast_kaggriculture/python/fast_kaggriculture" \
  -maxdepth 1 -name '_fast_kaggriculture*.so' -print -quit)
if [[ -z "$FAST_SO" ]]; then
  echo "missing built fast_kaggriculture extension" >&2
  exit 1
fi

mkdir -p "$HERE/build"
read -r -a PY_INCLUDES <<<"$($PYTHON_BIN -m pybind11 --includes)"
"$CXX_BIN" "${OPT_FLAGS[@]}" -std=c++20 -fPIC -shared \
  "${PY_INCLUDES[@]}" -I"$ROOT/fast_kaggriculture/src" \
  "$HERE/salemali7_2900.cpp" "$HERE/bindings.cpp" \
  -Wl,--no-as-needed "$FAST_SO" -Wl,-rpath,"$(dirname "$FAST_SO")" \
  -o "$HERE/build/salemali7_2900_native$EXT_SUFFIX"

echo "$HERE/build/salemali7_2900_native$EXT_SUFFIX"
