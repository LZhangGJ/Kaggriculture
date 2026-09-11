#!/usr/bin/env bash
set -euo pipefail

ROUTE_CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUTE_CONFIG="${1:-${ROUTE_CODE_ROOT}/configs/pipeline.env}"
if [[ ! -f "${ROUTE_CONFIG}" ]]; then
  echo "Missing config: ${ROUTE_CONFIG}" >&2
  echo "Copy configs/pipeline.env.example and fill its input paths." >&2
  exit 2
fi
# shellcheck source=/dev/null
source "${ROUTE_CONFIG}"

: "${REPLAY_MANIFEST:?set REPLAY_MANIFEST}"
: "${REPLAY_ROOT:?set REPLAY_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
if [[ -n "${EXTRA_REPLAY_MANIFEST:-}" && -z "${EXTRA_REPLAY_ROOT:-}" ]] || \
   [[ -z "${EXTRA_REPLAY_MANIFEST:-}" && -n "${EXTRA_REPLAY_ROOT:-}" ]]; then
  echo "Set both EXTRA_REPLAY_MANIFEST and EXTRA_REPLAY_ROOT, or neither." >&2
  exit 2
fi

ROUTE_PYTHON_BIN="${PYTHON_BIN:-python}"
ROUTE_FEATURE_WORKERS="${FEATURE_WORKERS:-12}"
ROUTE_AUDIT_WORKERS="${AUDIT_WORKERS:-96}"
ROUTE_TREE_WORKERS="${TREE_WORKERS:-14}"
export PYTHONPATH="${ROUTE_CODE_ROOT}/src:${ROUTE_CODE_ROOT}/fast_kaggriculture/python${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUTPUT_ROOT}/bin" "${OUTPUT_ROOT}/features" "${OUTPUT_ROOT}/agent"
if [[ ! -x "${OUTPUT_ROOT}/bin/intent_distance" ]]; then
  PYTHON_BIN="${ROUTE_PYTHON_BIN}" \
    "${ROUTE_CODE_ROOT}/scripts/build_native.sh" "${OUTPUT_ROOT}"
fi

"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/analyze_macro_route_library.py" \
  --manifest "${REPLAY_MANIFEST}" \
  --output "${OUTPUT_ROOT}/features/input-report.json" \
  --cache "${OUTPUT_ROOT}/features/input-features.npz" \
  --workers "${ROUTE_FEATURE_WORKERS}"
ROUTE_CACHE_ARGS=(--input "${OUTPUT_ROOT}/features/input-features.npz")
ROUTE_REPLAY_ARGS=(--replay-root "${REPLAY_ROOT}")
if [[ -n "${EXTRA_REPLAY_MANIFEST:-}" ]]; then
  "${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/analyze_macro_route_library.py" \
    --manifest "${EXTRA_REPLAY_MANIFEST}" \
    --output "${OUTPUT_ROOT}/features/extra-report.json" \
    --cache "${OUTPUT_ROOT}/features/extra-features.npz" \
    --workers "${ROUTE_FEATURE_WORKERS}"
  ROUTE_CACHE_ARGS+=(--input "${OUTPUT_ROOT}/features/extra-features.npz")
  ROUTE_REPLAY_ARGS+=(--replay-root "${EXTRA_REPLAY_ROOT}")
fi
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/merge_macro_route_caches.py" \
  "${ROUTE_CACHE_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/features-v1.npz" \
  --summary "${OUTPUT_ROOT}/merge-summary-v1.json"

"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/audit_macro_execution_quality.py" \
  --features "${OUTPUT_ROOT}/features-v1.npz" \
  "${ROUTE_REPLAY_ARGS[@]}" \
  --output "${OUTPUT_ROOT}/execution-audit-v1.npz" \
  --summary "${OUTPUT_ROOT}/execution-audit-summary-v1.json" \
  --workers "${ROUTE_AUDIT_WORKERS}"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/cluster_intended_macro_routes.py" \
  --features "${OUTPUT_ROOT}/features-v1.npz" \
  --audit "${OUTPUT_ROOT}/execution-audit-v1.npz" \
  --cpp-executable "${OUTPUT_ROOT}/bin/intent_distance" \
  --output "${OUTPUT_ROOT}/intent-clusters-v1.npz" \
  --summary "${OUTPUT_ROOT}/intent-clusters-summary-v1.json" \
  --threshold 0.12
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/export_macro_intent_route_specs.py" \
  --features "${OUTPUT_ROOT}/features-v1.npz" \
  --audit "${OUTPUT_ROOT}/execution-audit-v1.npz" \
  --clusters "${OUTPUT_ROOT}/intent-clusters-v1.npz" \
  --distance "${OUTPUT_ROOT}/intent-distance-v1.f32" \
  --output "${OUTPUT_ROOT}/macro-intent-routes-v1.json" \
  --drop-zero-win
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/export_intent_route_carriers.py" \
  --specs "${OUTPUT_ROOT}/macro-intent-routes-v1.json" \
  "${ROUTE_REPLAY_ARGS[@]}" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib"

"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/run_native_intent_round_robin.py" \
  --source "${ROUTE_CODE_ROOT}/runtime/teammate_base.py" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --seeds 410000:410064 \
  --output "${OUTPUT_ROOT}/intent-175-round-robin-64seeds-v1.npz" \
  --summary "${OUTPUT_ROOT}/intent-175-round-robin-64seeds-summary-v1.json"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/solve_route_nash.py" \
  --payoffs "${OUTPUT_ROOT}/intent-175-round-robin-64seeds-v1.npz" \
  --output "${OUTPUT_ROOT}/intent-175-nash-64seeds-v1.json"

ROUTE_OPENINGS="${OPENING_FAMILIES:-$("${ROUTE_PYTHON_BIN}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(x["family"] for x in d["opening_support"]))' \
  "${OUTPUT_ROOT}/intent-175-nash-64seeds-v1.json")}"
ROUTE_FINAL_OPENING="${FINAL_OPENING:-${ROUTE_OPENINGS%%,*}}"

"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/run_native_intent_switch_search.py" \
  --source "${ROUTE_CODE_ROOT}/runtime/teammate_base.py" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --openings "${ROUTE_OPENINGS}" \
  --checkpoints 48,72,96,120,144,168,216,264,312,360,408,456,504,576 \
  --seeds 420000:420008 \
  --output "${OUTPUT_ROOT}/intent-175-switch-coarse-8seeds-v1.npz" \
  --summary "${OUTPUT_ROOT}/intent-175-switch-coarse-8seeds-summary-v1.json"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/analyze_compact_switch_search.py" \
  --input "${OUTPUT_ROOT}/intent-175-switch-coarse-8seeds-v1.npz" \
  --output "${OUTPUT_ROOT}/intent-175-switch-coarse-analysis-v1.json" \
  --min-greedy-gain 0.0005

ROUTE_FINE_TARGETS="$("${ROUTE_PYTHON_BIN}" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); print(",".join(d["screening"]["selected_families"]))' \
  "${OUTPUT_ROOT}/intent-175-switch-coarse-analysis-v1.json")"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/run_native_intent_switch_search.py" \
  --source "${ROUTE_CODE_ROOT}/runtime/teammate_base.py" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --openings "${ROUTE_OPENINGS}" \
  --targets "${ROUTE_FINE_TARGETS}" \
  --checkpoints 48,72,96,120,144,168,216 \
  --seeds 430000:430128 \
  --output "${OUTPUT_ROOT}/intent-11-switch-fine-128seeds-v1.npz" \
  --summary "${OUTPUT_ROOT}/intent-11-switch-fine-128seeds-summary-v1.json"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/train_robust_search_route_trees.py" \
  --search "${OUTPUT_ROOT}/intent-11-switch-fine-128seeds-v1.npz" \
  --output "${OUTPUT_ROOT}/intent-11-switch-robust-trees-128seeds-v1.json" \
  --depths 2,3,4 \
  --min-leaves 512,1024,2048 \
  --min-robust-improvement 0.01 \
  --simplicity-tolerance 0.005 \
  --node-workers "${ROUTE_TREE_WORKERS}"

"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/search_native_route_tree_sequences.py" \
  --source "${ROUTE_CODE_ROOT}/runtime/teammate_base.py" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --policy "${OUTPUT_ROOT}/intent-11-switch-robust-trees-128seeds-v1.json" \
  --opening "${ROUTE_FINAL_OPENING}" \
  --seeds 450000:450128 \
  --output "${OUTPUT_ROOT}/intent-11-switch-sequence-search-128seeds-v1.json" \
  --selected-policy-output "${OUTPUT_ROOT}/intent-11-switch-selected-policy-v1.json"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/search_native_route_tree_sequences.py" \
  --source "${ROUTE_CODE_ROOT}/runtime/teammate_base.py" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --policy "${OUTPUT_ROOT}/intent-11-switch-selected-policy-v1.json" \
  --opening "${ROUTE_FINAL_OPENING}" \
  --seeds 460000:460256 \
  --output "${OUTPUT_ROOT}/intent-11-switch-final-holdout-256seeds-v1.json" \
  --selected-policy-output "${OUTPUT_ROOT}/intent-11-switch-final-policy-v1.json" \
  --fixed-all

"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/export_teammate_meta_submission.py" \
  --base "${ROUTE_CODE_ROOT}/runtime/teammate_base.py" \
  --actions "${OUTPUT_ROOT}/route-actions.json.zlib" \
  --metadata "${OUTPUT_ROOT}/route-library.json" \
  --policy "${OUTPUT_ROOT}/intent-11-switch-final-policy-v1.json" \
  --nash "${OUTPUT_ROOT}/intent-175-nash-64seeds-v1.json" \
  --holdout "${OUTPUT_ROOT}/intent-11-switch-final-holdout-256seeds-v1.json" \
  --forced-opening "${ROUTE_FINAL_OPENING}" \
  --output "${OUTPUT_ROOT}/agent/multifile"
"${ROUTE_PYTHON_BIN}" "${ROUTE_CODE_ROOT}/scripts/export_single_file_submission.py" \
  --source-dir "${OUTPUT_ROOT}/agent/multifile" \
  --forced-opening "${ROUTE_FINAL_OPENING}" \
  --output "${OUTPUT_ROOT}/agent/main.py"

echo "pipeline_status=OK"
echo "final_agent=${OUTPUT_ROOT}/agent/main.py"
