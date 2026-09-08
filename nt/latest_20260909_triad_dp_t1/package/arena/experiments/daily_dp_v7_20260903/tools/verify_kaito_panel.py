"""Final stage receipt: port fidelity, isolated memory, unchanged old panels."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import statistics

EXP=Path(__file__).resolve().parents[1]
def read(rel):return json.loads((EXP/rel).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def signature(row):
    fields=("variant","opponent","seed","seat","steps","cash","opponent_cash","win","margin",
            "opponent_switched","overflow","selected_segments","opponent_state","weed_repair_frames")
    return json.dumps({k:row.get(k) for k in fields},sort_keys=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    build=read("native/build/build_receipt.json")
    binary=next((EXP/"native/build").glob("_dp7_native*.so"));assert sha(binary)==build["binary_sha256"]
    for rel,h in build["source_hashes"].items():assert sha(EXP/rel)==h
    reg=read("opponents/registry.json");entry=reg["opponents"]["kaito_v58"]
    assert sha(EXP/entry["source"])==entry["source_sha256"] and sha(EXP/entry["asset"])==entry["asset_sha256"]
    checks={k:read(entry[k]) for k in ("initial_parity_receipt","mirror_parity_receipt","router_receipt","isolation_receipt")}
    for check in checks.values():assert check["status"]=="PASS" and check["build"]["binary_sha256"]==build["binary_sha256"]
    assert checks["router_receipt"]["covered_output_routes"]==list(range(10))
    assert checks["isolation_receipt"]["parallel_repeat_games"]==1000
    official=[r for k in ("initial_parity_receipt","mirror_parity_receipt") for r in checks[k]["rows"]]
    for r in official:
        assert r["steps"]==719 and r["action_mismatches"]==r["memory_mismatches"]==r["official_state_mismatches"]==0
    assert any(any(r["collisions_by_route"]) for r in official)
    assert any(any(r["preempt_units_by_route"]) for r in official)
    for opponent in ("boatlee_v29","yhay81_six_day"):
        for key in ("initial_parity_receipt","isolation_receipt"):
            r=read(reg["opponents"][opponent][key]);assert r["status"]=="PASS" and r["build"]["binary_sha256"]==build["binary_sha256"]
    mechanisms=read("receipts/kaito_unchanged_policy_mechanisms_v1/acceptance.json")
    assert mechanisms["status"]=="PASS" and mechanisms["native_mechanism_checks"]==546
    preserved=0;panels=[];fingerprints=[]
    for part in ("A","B"):
        before=read(f"receipts/pool_boatlee_{part}50_v1/results.json")
        after=read(f"receipts/pool_kaito_{part}50_v1/results.json")
        assert after["status"]=="COMPLETE_PANEL_NOT_GOAL_ACCEPTANCE"
        assert after["build"]["binary_sha256"]==build["binary_sha256"]
        old_rows=[r for r in after["rows"] if r["opponent"]!="kaito_v58"]
        assert [signature(r) for r in old_rows]==[signature(r) for r in before["rows"]]
        preserved+=len(old_rows);panels.append(after)
        for rel in ("native/policy.hpp","native/vendor/simulator.cpp","native/vendor/simulator.hpp",
                    "native/vendor/native_teammate.cpp","native/vendor/native_teammate.hpp",
                    "native/boatlee_v29.cpp","native/boatlee_v29.hpp","native/fieldbook_adapter.cpp","native/fieldbook_adapter.hpp"):
            assert before["build"]["source_hashes"][rel]==after["build"]["source_hashes"][rel],rel
    lookup={(r["seed"],r["seat"]):r for r in panels[0]["rows"] if r["opponent"]=="kaito_v58" and r["variant"]=="L3_base"}
    for ref in checks["initial_parity_receipt"]["rows"]:
        actual=lookup[ref["seed"],ref["seat"]]
        for key in ("cash","opponent_cash","selected_route_frames","preempt_units_by_route"):
            assert actual[key]==ref[key]
        assert actual["weed_collisions_by_route"]==ref["collisions_by_route"]
    aggregate={}
    for variant in panels[0]["summary"]:
        unique={(r["seed"],r["seat"]):r for p in panels for r in p["rows"] if r["variant"]==variant and r["opponent"]=="kaito_v58"}
        clusters=[(int(unique[s,0]["win"])+int(unique[s,1]["win"]))/2 for s in sorted({s for s,_ in unique})]
        rate=statistics.fmean(clusters);se=statistics.stdev(clusters)/math.sqrt(len(clusters))
        aggregate[variant]=dict(games=len(unique),seeds=len(clusters),wins=sum(r["win"] for r in unique.values()),
                                win_rate=rate,seed_cluster_se=se,approximate_95pct_interval=[max(0,rate-1.96*se),min(1,rate+1.96*se)],
                                mean_cash=statistics.fmean(r["cash"] for r in unique.values()),
                                mean_opponent_cash=statistics.fmean(r["opponent_cash"] for r in unique.values()),
                                mean_margin=statistics.fmean(r["margin"] for r in unique.values()))
    receipt=dict(status="PASS_ADAPTER_AND_REGRESSION_NOT_GOAL_ACCEPTANCE",build=build,
                 frozen_candidate_rules_and_old_opponents_unchanged=True,old_panel_rows_unchanged=preserved,
                 current_build_official_games=len(official),official_steps=len(official)*719,
                 synthetic_router_cases=checks["router_receipt"]["cases"],parallel_repeat_games=1000,
                 aggregate=aggregate,native_opponents=[k for k,v in reg["opponents"].items() if v["runtime"]!="pending_native"],
                 pending_opponents=[k for k,v in reg["opponents"].items() if v["runtime"]=="pending_native"],
                 final_goal_acceptance=False,final_holdout_used=False,
                 caveat="Development seeds and finite parity. No new candidate promoted, no claim of all-state equivalence or 90% wins.")
    (out/"acceptance.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!="build"}),flush=True)


if __name__=="__main__":main()
