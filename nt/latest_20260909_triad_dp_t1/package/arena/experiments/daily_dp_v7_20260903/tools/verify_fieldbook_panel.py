"""Reconcile added-opponent panels with old results and parity receipts."""
from pathlib import Path
import argparse
import hashlib
import json

EXP = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((EXP / rel).read_text())


def signatures(rows):
    fields = ("variant", "opponent", "seed", "seat", "cash", "opponent_cash", "opponent_switched")
    return [tuple(r[f] for f in fields) for r in rows]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    old = read("receipts/pool_g001_g003_A50_v2/results.json")
    first = read("receipts/pool_fieldbook_A50_v1/results.json")
    second = read("receipts/pool_fieldbook_B50_v1/results.json")
    assert signatures(old["rows"]) == signatures([r for r in first["rows"] if r["opponent"] != "yhay81_six_day"])
    checks = [read("receipts/" + name + "/acceptance.json") for name in
              ("fieldbook_official_parity_v1", "fieldbook_official_branches_v1", "fieldbook_context_isolation_v1")]
    binary = first["build"]["binary_sha256"]
    assert all(c["status"] == "PASS" and c["build"]["binary_sha256"] == binary for c in checks)
    lookup = {(r["variant"], r["opponent"], r["seed"], r["seat"]): r for r in first["rows"]}
    official_rows = checks[0]["rows"] + checks[1]["rows"]
    for ref in official_rows:
        row = lookup["L3_base", "yhay81_six_day", ref["seed"], ref["seat"]]
        for field in ("cash", "opponent_cash", "selected_segments"):
            assert row[field] == ref[field]
    registry = read("opponents/registry.json")
    for panel in (first, second):
        assert panel["status"] == "COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE"
        assert all(r["steps"] == 719 for r in panel["rows"])
        for k in ("native/policy.hpp", "native/vendor/simulator.cpp", "native/vendor/simulator.hpp",
                  "native/vendor/native_teammate.cpp", "native/vendor/native_teammate.hpp"):
            assert panel["build"]["source_hashes"][k] == old["build"]["source_hashes"][k]
    receipt = dict(status="PASS_ADAPTER_AND_REGRESSION_NOT_GOAL_ACCEPTANCE",
                   binary_sha256=binary, prior_A_panel_rows_unchanged=len(old["rows"]),
                   original_policy_and_rule_engine_hashes_unchanged=True,
                   official_games=len(official_rows), official_steps=len(official_rows)*719,
                   projected_fields=sum(r["projection_fields_checked"] for r in official_rows),
                   official_batch_terminal_parity=True, threaded_repeat_games=1000,
                   implemented_real_opponents=[k for k, e in registry["opponents"].items() if e["runtime"] != "pending_native"],
                   pending_real_opponents=[k for k, e in registry["opponents"].items() if e["runtime"] == "pending_native"],
                   registry_sha256=hashlib.sha256((EXP / "opponents/registry.json").read_bytes()).hexdigest(),
                   summary_A=first["summary"], summary_B=second["summary"])
    (out / "acceptance.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if not k.startswith("summary_")}), flush=True)


if __name__ == "__main__":
    main()
