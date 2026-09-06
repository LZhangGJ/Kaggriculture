#!/usr/bin/env bash
# Package-prescribed local validation: smoke -> portable parity -> evaluate.
set -e
export OMP_NUM_THREADS=1
B="$HOME/kag/nt/latest_20260906_c3_f3_j7c3_search"
K="$HOME/kag/nt/latest_20260906_keep2_rl_1000"
PY="$B/.venv/bin/python"

echo "##### 1. smoke.py (base package delivery/parity checks) #####"
cd "$B"
rm -rf runs/smoke_first
date
"$PY" smoke.py --out runs/smoke_first 2>&1 | tail -15
date
echo

echo "##### 2. validate_portable.py (812 games vs frozen receipts) #####"
cd "$K"
date
"$PY" validate_portable.py 2>&1 | tail -15
date
"$PY" - <<'EOF'
import json
r=json.load(open("PORTABLE_ACCEPTANCE.json"))
print("status",r["status"],"games",r["games"],"configs",r["configs"],
      "checks_all_exact",all(v["cash_win_and_execution_counters_exact"] for v in r["checks"].values()))
EOF
echo

echo "##### 3. evaluate.py keep / aux_r0 greedy (2 seeds x 7 opp x 2 seats) #####"
rm -rf runs/keep runs/aux1000_greedy
"$PY" evaluate.py --mode keep --out runs/keep 2>&1 | tail -30
echo "-----"
"$PY" evaluate.py --out runs/aux1000_greedy 2>&1 | tail -30
echo VALIDATION_CHAIN_DONE
