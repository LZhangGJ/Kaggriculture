"""Synthetic public branch fixtures compared to the untouched V58 router body."""
from pathlib import Path
import argparse
import ast
import copy
import itertools
import json
import sys
import zlib

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/"native/build"))
sys.path.insert(0,str(EXP.parents[1]/"gpt_review/codex/G001_CPU_FOR_GPT_20260903"))
import _dp7_native as native
from cpu_runtime import load_agent
from check_native import canon
from check_boatlee_native import SHOPS, digest


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    source=EXP/"opponents/kaito_v58/output/main.py"
    data=json.loads(zlib.decompress((EXP/"native/kaito_v58_frozen.json.zlib").read_bytes()))
    assert digest(source)==data["source_sha256"]
    agent=native.KaitoV58(data);reference=load_agent(source);ns=reference.__globals__
    # Replace only child outputs with markers. Keep original router unchanged.
    ns["_v58_advance_extra_policies"]=lambda obs,config:{r["name"]:{"marker":i} for i,r in enumerate(data["routes"])}
    tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_v58_recovery_mode")
    signatures=[]
    for node in ast.walk(fn):
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=="signature":
            signatures.append(ast.literal_eval(node.comparators[0]))
    assert len(signatures)==8
    cases=[]
    for sig in signatures:
        first=SHOPS.index(sig[0]);nums=list(sig[1:])
        cases.append(dict(step=72,shops=[first],signature=nums))
        for coordinate in range(7):
            for delta in (-1,1):
                changed=nums.copy();changed[coordinate]+=delta
                cases.append(dict(step=72,shops=[first],signature=changed))
    for shops,step in itertools.product([[],*[[i] for i in range(8)],*[[i,j] for i in range(8) for j in range(8)]],[0,71,72,96,143,144,360,718]):
        cases.append(dict(step=step,shops=shops))
    modes=[None,"base","farmers-recovery","recovery","ice-minimax","bakery-second-shop","pizza-second-shop"]
    for mode,second,step in itertools.product(modes,range(8),(143,144,360,718)):
        cases.append(dict(step=step,shops=[7,second],state=dict(mode=mode,known_yarn=True,mirror_streak=250,clone_selected=True)))
    for sig in ((214,214,9982,4,2,7,12),(215,125,9979,4,2,7,12)):
        cases.append(dict(step=96,shops=[7],signature=sig))
        for coordinate in range(7):
            for delta in (-1,1):
                altered=list(sig);altered[coordinate]+=delta
                cases.append(dict(step=96,shops=[7],signature=altered))
    for step,streak,mirror in itertools.product((359,360,361),(238,239,240),(False,True)):
        cases.append(dict(step=step,shops=[7],mirror=mirror,state=dict(mode="base",mirror_streak=streak)))
    rows=[]
    for spec in cases:
        result=canon(native.kaito_router_probe(agent,spec));state=dict(last_step=spec["step"]-1,mode=None,mirror_streak=0,clone_selected=False,known_yarn=False)
        state.update(spec.get("state",{}));ns["_V58_STATES"][0]=state
        expected=reference(copy.deepcopy(result["observation"]),data["official_configuration"])
        actual=result["state"];selected=actual.pop("selected");actual.pop("policies")
        ref=canon(ns["_V58_STATES"][0])
        if expected.get("marker")!=selected or actual!=ref:
            (out/"failure.json").write_text(json.dumps(dict(spec=spec,actual=actual,expected=ref,selected=selected,expected_action=expected),indent=2))
            raise AssertionError(spec)
        rows.append(dict(spec=spec,selected=selected,state=actual))
    receipt=dict(status="PASS",build=json.loads((EXP/"native/build/build_receipt.json").read_text()),
                 source_sha256=digest(source),cases=len(rows),rows=rows,mismatches=0,
                 covered_output_routes=sorted(set(r["selected"] for r in rows)),
                 caveat="Synthetic public router unit tests, not natural match outcomes or source strength evidence.")
    (out/"acceptance.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("build","rows")}),flush=True)


if __name__=="__main__":main()
