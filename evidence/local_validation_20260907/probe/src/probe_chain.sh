#!/usr/bin/env bash
# agent.md section 9.6 step 0: compile the dV probe library and run it against frozen KEEP.
set -e
export OMP_NUM_THREADS=1
SP="/mnt/c/Users/DHU_Z/AppData/Local/Temp/claude/d--Kaggriculture-keep2-rl-1000/748d2719-6a93-4579-8d30-be90143373d3/scratchpad"
B="$HOME/kag/nt/latest_20260906_c3_f3_j7c3_search"
PY="$B/.venv/bin/python"
cd "$B"
tr -d '\r' < "$SP/f3_deltav_policy.cpp" > src/f3_deltav_policy.cpp
tr -d '\r' < "$SP/deltav_probe.py" > deltav_probe.py

echo "##### diff vs frozen f3_policy.cpp (only PROBE lines should differ) #####"
diff src/f3_policy.cpp src/f3_deltav_policy.cpp || true
echo

echo "##### compile f3_deltav.so (same flags as build.py f3 arm) #####"
date
g++ -std=c++20 -O3 -DNDEBUG -fPIC -ffp-contract=off -pthread -shared -Wl,-Bsymbolic \
  -I"$B/vendor" src/f3_deltav_policy.cpp -o build/f3_deltav.so
date
ls -la build/f3_deltav.so
echo

echo "##### probe: 100 seeds x 7 opponents x 2 seats = 1400 games per config #####"
rm -rf runs/deltav_probe_100
"$PY" deltav_probe.py --seeds 100 --threads 16 --out runs/deltav_probe_100
echo PROBE_CHAIN_DONE
