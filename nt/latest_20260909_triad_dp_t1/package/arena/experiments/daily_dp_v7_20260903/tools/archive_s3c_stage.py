"""Immutable stage receipt and frozen source copies, not a goal acceptance."""
from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1];out=EXP/'profiles/s3c'
receipt=out/'stage_receipt.json';assert not receipt.exists()
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    src=EXP/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'source_snapshot'/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst)
for rel in ('native/test_policy.cpp','tools/run_production_audit.py','tools/prepare_s3c_land.py','tools/prepare_s3c_capacity.py','tools/prepare_s3c_hauling.py','tools/prepare_s3c_feed_commitments.py'):
    dst=out/'source_snapshot'/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(EXP/rel,dst)
source_expectations={
 'gpt_review/gpt_code/agent_v6_170k.py':'5ffd91a3091f64282984cab47ca8dc913e47376b4643e628397c3afc8d8143ba',
 'gpt_review/gpt_code/my_agent_v7.py':'8147655aa8a3d0bee3a130352272488143c6d0e89decd922cebbe9a4a2825db6',
 'research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py':'9b4fdf7a7c92d3eefc75c693c4949825567588478bef0e53529bdec82944dda7'}
for rel,h in source_expectations.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
names=['s3c_land_A','s3c_land_B','s3c_hauling_A','s3c_hauling_B','s3c_capacity_A']
results={name:json.loads((EXP/'receipts'/name/'results.json').read_text()) for name in names}
current=results['s3c_hauling_A'];prior=results['s3c_land_A']
def cash_rows(data,label):
    return {(r['opponent'],r['seed'],r['seat']):(r['cash'],r['opponent_cash']) for r in data['rows'] if r['label']==label}
assert cash_rows(current,'L3_base')==cash_rows(prior,'S3C03')
assert cash_rows(current,'L4_base')==cash_rows(prior,'allow_four')
checks={}
for name in ('s3c_commitment_mechanisms','s3c_return_official','s3c_four_hauling_official'):
    d=json.loads((EXP/'receipts'/name/'acceptance.json').read_text());assert d['status']=='PASS';checks[name]=d
land_data=json.loads((EXP/'receipts/s3b_cash_audit_dev50/S3C03_g001.json').read_text())
lands=[1+sum(day[1-row['seat']]['lands'] for day in row['days']) for row in land_data['rows']]
assert len(lands)==100 and set(lands)=={3}
profiles=json.loads((out/'hauling_configs.json').read_text())
(out/'S3C03R_provisional.json').write_text(json.dumps(profiles['L3_return'],indent=2),encoding='utf-8')
record=dict(status='DEVELOPMENT_NOT_GOAL_ACCEPTED',build=build,original_sources=source_expectations,
    summaries={k:v['summaries'] for k,v in results.items()},checks=checks,
    default_best_unchanged='profiles/S3C03.json',conditional_candidate='profiles/s3c/S3C03R_provisional.json',
    regression='L3 and L4 control per-game cash unchanged after adding default-off hauling',
    g001_land_distribution_in_existing_A100={'3':100},
    g003={'submission':55918053,'source_identity_verified':True,'source_available':False,'integrated':False},
    feed_commitments='mechanism tested; no strength test yet',
    final_holdouts_used=False,full_goal_complete=False)
receipt.write_text(json.dumps(record,indent=2),encoding='utf-8');print(receipt)
