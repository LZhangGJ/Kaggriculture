from pathlib import Path
import hashlib,json,shutil
E=Path(__file__).resolve().parents[1];out=E/'profiles/s4u';out.mkdir(exist_ok=False)
b=json.loads((E/'native/build/build_receipt.json').read_text());old=E/'receipts/s4t_fiveway_N50_v1/results.json'
partial=json.loads(old.read_text());assert partial['status']=='RUNNING' and len(partial['rows'])==3600
for rel,h in b['source_hashes'].items():
    src=E/rel;assert hashlib.sha256(src.read_bytes()).hexdigest()==h
    dst=out/'before'/rel;dst.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(src,dst)
shutil.copy2(old,out/'s4t_partial_results.json')
cfg=json.loads((E/'profiles/s4t/configs.json').read_text())
for label,p in cfg.items():
    p['exact_schedule_cache']=label.startswith('full_');p['incremental_regret_cost']=label.startswith('full_')
(out/'configs.json').write_text(json.dumps(cfg,indent=2))
(out/'input.json').write_text(json.dumps(dict(status='FROZEN_INPUT_BEFORE_IMPLEMENTATION',before_build=b,
    old_completed_games=3600,old_panel_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
    configs_sha256=hashlib.sha256((out/'configs.json').read_bytes()).hexdigest(),holdout_used=False,
    old_execution='Stopped deliberately for performance replacement; PIDs 291/563 absent after TERM/CONT, handle4775 exit1. Not a rule failure.'),indent=2))
(E/'receipts/s4t_execution_v1/interruption.json').write_text(json.dumps(dict(status='INTERRUPTED_FOR_EXACT_PERFORMANCE_UPGRADE',
    completed_games=3600,uncompleted_variant='full_chain_autonomous',source_hashes=b['source_hashes'],
    successor='profiles/s4u',old_records_preserved=True,goal_complete=False),indent=2))
print('FROZEN_S4U_INPUT',flush=True)
