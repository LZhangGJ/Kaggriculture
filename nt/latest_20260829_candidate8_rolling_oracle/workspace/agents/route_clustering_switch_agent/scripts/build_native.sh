#!/usr/bin/env bash
set -euo pipefail

# The extension imports pybind11 from the selected environment in setup.py.
# Disable pip build isolation so that the pinned, known-compatible pybind11 is
# used instead of an empty temporary build environment.

ROUTE_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUTE_PYTHON_BIN="${PYTHON_BIN:-python}"
ROUTE_BUILD_ROOT="${1:-${ROUTE_CODE_ROOT}/build}"

mkdir -p "${ROUTE_BUILD_ROOT}/bin"
"${ROUTE_PYTHON_BIN}" -m pip install --no-build-isolation -e "${ROUTE_CODE_ROOT}/fast_kaggriculture"
c++ -O3 -DNDEBUG -std=c++20 -fopenmp \
  "${ROUTE_CODE_ROOT}/fast_kaggriculture/tools/intent_distance.cpp" \
  -o "${ROUTE_BUILD_ROOT}/bin/intent_distance"

"${ROUTE_PYTHON_BIN}" -c \
  'from fast_kaggriculture import NativeTeammateExecutor; print("native_extension=OK")'
echo "intent_distance=${ROUTE_BUILD_ROOT}/bin/intent_distance"
