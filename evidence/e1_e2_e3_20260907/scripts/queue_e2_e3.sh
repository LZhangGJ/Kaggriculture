#!/usr/bin/env bash
# Waits for the already-running, detached E1 job (PID tracked via pgrep on its
# argv) to finish, then runs E2 (3 held-out opponents x 2 repeats each) and E3
# (3 repeats) sequentially. Single shared GPU, so these cannot overlap E1 or
# each other; this queue removes the need to babysit each stage manually.
set -e
B="$HOME/kag/nt/latest_20260906_c3_f3_j7c3_search"
PY="$B/.venv/bin/python"

echo "QUEUE: waiting for E1 (e1_train.py) to finish..."
while pgrep -f 'e1_train.py' > /dev/null; do sleep 30; done
echo "QUEUE: E1 finished at $(date)"
if [ -f "$HOME/kag/e1_training/aux_SUMMARY_SO_FAR.json" ]; then
    echo "QUEUE: E1 summary present"
else
    echo "QUEUE: WARNING E1 summary missing, check for crash before trusting E2/E3 GPU availability assumption"
fi

cd "$HOME/kag/e2_scripts"
export OMP_NUM_THREADS=1
for held in three_day g003 kaito; do
    echo "QUEUE: starting E2 held-out=$held at $(date)"
    "$PY" -u e2_train.py --held-out "$held" --repeats 2 --rounds 300
    echo "QUEUE: finished E2 held-out=$held at $(date)"
done

cd "$HOME/kag/e3_scripts"
echo "QUEUE: starting E3 at $(date)"
"$PY" -u e3_train.py --repeats 3 --rounds 300
echo "QUEUE: finished E3 at $(date)"

echo "QUEUE_ALL_DONE $(date)"
