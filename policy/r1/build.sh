#!/usr/bin/env bash
set -euo pipefail
stable=$(cd "$(dirname "$0")" && pwd)
cd "$stable"
g++ -std=c++20 -O3 -DNDEBUG -shared -fPIC -pthread \
  -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 \
  -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=0 \
  -DR2_SALE_CLOCK_MODE=0 -DP16_WORKING_CAPITAL_GATE=1 -DP16_LIVE_REMAINING_VALUE=1 \
  -DT3_OBLIGATION_REPAIR=1 -DT3_CAPACITY_REPAIR=1 -DT3_RECEIPT_REPAIR=1 \
  -DP16_JOINT_BUNDLES=1 -I"$stable" bridge.cpp "$stable/executor/vendor/simulator.cpp" -o agent.so
