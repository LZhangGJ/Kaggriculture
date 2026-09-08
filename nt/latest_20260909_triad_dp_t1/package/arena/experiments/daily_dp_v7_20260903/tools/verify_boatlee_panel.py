"""Verify current Boatlee source, original parity, old-result preservation."""
from pathlib import Path
import argparse
import hashlib
import json

EXP=Path(__file__).resolve().parents[1]


def read(rel):return json.loads((EXP/rel).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def signature(row):
    fields=("variant","opponent","seed","seat","steps","cash","opponent_cash","opponent_switched","overflow","selected_segments")
    return json.dumps({k:row.get(k) for k in fields},sort_keys=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    build=read("native/build/build_receipt.json")
    for rel,h in build["source_hashes"].items():assert sha(EXP/rel)==h
    registry=read("opponents/registry.json");entry=registry["opponents"]["boatlee_v29"]
    assert sha(EXP/entry["source"])==entry["source_sha256"]
    assert sha(EXP/entry["asset"])==entry["asset_sha256"]
    checks=[read(entry[k]) for k in ("initial_parity_receipt","weed_parity_receipt","mirror_parity_receipt","isolation_receipt")]
    assert all(c["status"]=="PASS" and c["build"]["binary_sha256"]==build["binary_sha256"] for c in checks)
    official=[r for c in checks[:3] for r in c["rows"]]
    assert any(r["opponent_state"]["near_mirror"]==1 for r in official)
    assert any(r["weed_repair_frames"]>0 for r in official)
    assert any(r["adaptive_market_frames"]>0 for r in official)
    total=0;panels=[]
    for part in ("A","B"):
        before=read(f"receipts/pool_fieldbook_{part}50_v1/results.json")
        after=read(f"receipts/pool_boatlee_{part}50_v1/results.json")
        assert after["status"]=="COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE"
        selected=[r for r in after["rows"] if r["opponent"]!="boatlee_v29"]
        assert [signature(r) for r in before["rows"]]==[signature(r) for r in selected]
        total+=len(selected);panels.append(after)
        for rel in ("native/policy.hpp","native/vendor/simulator.cpp","native/vendor/simulator.hpp",
                    "native/vendor/native_teammate.cpp","native/vendor/native_teammate.hpp"):
            assert before["build"]["source_hashes"][rel]==after["build"]["source_hashes"][rel]
    lookup={(r["seed"],r["seat"]):r for r in panels[0]["rows"] if r["variant"]=="L3_base" and r["opponent"]=="boatlee_v29"}
    for c in checks[:2]:
        for ref in c["rows"]:
            actual=lookup[ref["seed"],ref["seat"]]
            for k in ("cash","opponent_cash","opponent_state","weed_repair_frames"):assert actual[k]==ref[k]
    receipt=dict(status="PASS_ADAPTER_AND_REGRESSION_NOT_GOAL_ACCEPTANCE",build=build,
                 original_rule_and_candidate_hashes_unchanged=True,old_panel_rows_unchanged=total,
                 official_games=len(official),official_steps=len(official)*719,
                 official_original_actions_and_memory_mismatches=0,official_state_mismatches=0,
                 mirror_and_weed_and_adaptive_market_coverage=True,
                 parallel_repeat_games=checks[3]["parallel_repeat_games"],
                 native_opponents=[k for k,e in registry["opponents"].items() if e["runtime"]!="pending_native"],
                 pending_opponents=[k for k,e in registry["opponents"].items() if e["runtime"]=="pending_native"],
                 summary_A=panels[0]["summary"],summary_B=panels[1]["summary"])
    (out/"acceptance.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!="build" and not k.startswith("summary_")}),flush=True)


if __name__=="__main__":main()
