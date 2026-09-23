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
[[ -n "$FAST_SO" ]] || { echo "missing built fast_kaggriculture extension" >&2; exit 1; }

mkdir -p "$HERE/build"
read -r -a PY_INCLUDES <<<"$($PYTHON_BIN -m pybind11 --includes)"
INCLUDES=(
  -I"$ROOT/fast_kaggriculture/src"
  -I"$ROOT/native_deps/findNewRoad/repairMechanismCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/productionObligationCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/productionForecastCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/legacyBaselineForecastCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/plannerInputCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/economicIntentBridgeCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/protectedQueueCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/queueInvariantProofCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/robustCertificateCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/phaseInputCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/phasedTakeoverCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/publicBeliefRuntimeCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/nativeObservationAdapterCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/nativeSelectiveInputCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/selectiveRuntimeCpp/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/general_econ_audit/include"
  -I"$ROOT/native_deps/findNewRoad/marketMechanismCpp/include"
)

"$CXX_BIN" "${OPT_FLAGS[@]}" -std=c++20 -fopenmp -fPIC -shared \
  "${PY_INCLUDES[@]}" "${INCLUDES[@]}" \
  "$HERE/thomas_2945.cpp" "$HERE/bindings.cpp" \
  -Wl,--no-as-needed "$FAST_SO" \
  -Wl,-rpath,"$(dirname "$FAST_SO")" -fopenmp \
  -o "$HERE/build/thomas_2945_cpp_native$EXT_SUFFIX"

echo "$HERE/build/thomas_2945_cpp_native$EXT_SUFFIX"
