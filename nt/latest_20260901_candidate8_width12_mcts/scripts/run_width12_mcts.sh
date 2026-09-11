#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-width}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="${BUNDLE_ROOT}/workspace"
AGENT="${WORKSPACE}/agents/route_clustering_switch_agent"
EXPERIMENT="${WORKSPACE}/experiments/ecobot_adaptive_planner_v1"
FAST="${AGENT}/fast_kaggriculture"
OUTPUT_DIR="${BUNDLE_ROOT}/outputs"
PYTHON_BIN="${PYTHON_BIN:-python}"
MCTS_WORKERS="${MCTS_WORKERS:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONPATH="${AGENT}/src:${EXPERIMENT}/tools:${FAST}/python${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_DIR}"

if ! compgen -G "${FAST}/python/fast_kaggriculture/_fast_kaggriculture*.so" >/dev/null; then
  (
    cd "${FAST}"
    "${PYTHON_BIN}" setup.py build_ext --inplace
  )
fi

COMMON=(
  --source "${AGENT}/runtime/teammate_base.py"
  --actions "${EXPERIMENT}/artifacts/oracle/rank1_latest_55714246_ep99954642_route_actions.json.zlib"
  --metadata "${EXPERIMENT}/artifacts/oracle/rank1_latest_55714246_ep99954642_route_library.json"
  --genomes "${EXPERIMENT}/configs/w5_autonomous_capacity_r6_challenger_v1.json"
  --merged-receipt "${EXPERIMENT}/receipts/cxx_clean133_actual_vs_oracle_confirmed_v1.json"
  --opponents G001,G004,G013
  --seed-start 3101101
  --seed-count 2
  --decision-days 0,1,3,6,9,12,18,24
  --per-node-arms 64
  --candidate-pool shortlist
  --mcts-workers "${MCTS_WORKERS}"
)

case "${MODE}" in
  width)
    "${PYTHON_BIN}" "${EXPERIMENT}/tools/compare_candidate8_mcts_beam.py" \
      "${COMMON[@]}" \
      --beam-widths 2,4,8,12 \
      --mcts-repeats 0 \
      --output "${OUTPUT_DIR}/candidate8_beam_width_2_4_8_12.json"
    ;;
  mcts)
    "${PYTHON_BIN}" "${EXPERIMENT}/tools/compare_candidate8_mcts_beam.py" \
      "${COMMON[@]}" \
      --beam-widths 1,4 \
      --mcts-repeats 2 \
      --output "${OUTPUT_DIR}/candidate8_mcts_vs_beam.json"
    ;;
  *)
    echo "usage: $0 {width|mcts}" >&2
    exit 2
    ;;
esac
