#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
cxx=${CXX_BIN:-g++}
output=${1:-"$root/work/agent-student-actor-owned-v3.so"}
mkdir -p "$(dirname "$output")"

"$cxx" -std=c++20 -O3 -DNDEBUG -shared -fPIC -pthread \
  -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 \
  -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 \
  -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=0 \
  -DR2_SALE_CLOCK_MODE=0 -DP16_WORKING_CAPITAL_GATE=1 \
  -DP16_LIVE_REMAINING_VALUE=1 -DT3_OBLIGATION_REPAIR=1 \
  -DT3_CAPACITY_REPAIR=1 -DT3_RECEIPT_REPAIR=1 \
  -DP16_JOINT_BUNDLES=1 -DR2_STUDENT_SLOT_AUDIT=1 \
  -DR2_OUTER_NEIGHBOR_AUDIT=1 -DR2_OUTER_BEAM_WIDTH=2 \
  -DR2_STUDENT_MAX_LAND="${STUDENT_MAX_LAND:-3}" \
  -DR2_STUDENT_LIFECYCLE_V4="${STUDENT_LIFECYCLE_V4:-0}" \
  -I"$root/policy/r1" \
  "$root/policy/r1/teacher_bridge.cpp" \
  "$root/policy/r1/executor/vendor/simulator.cpp" \
  -o "$output"

echo "$output"
