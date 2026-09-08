"""Freeze before independent holdout; never tune against these results."""
from experiment import *
import subprocess
cfg=json.loads((R/'policy/config.json').read_text());p=pool()
# Validate final source changes on the SAME development panel, not the holdout.
run(p,'release_dev32',cfg,count=32)
receipt=json.loads((R/'policy/agent.build.json').read_text())
freeze=dict(name='Triad-DP T1',binary_hash=receipt['binary_hash'],source_hash=receipt['source_hash'],config=cfg,model_active=False,holdout_seed_start=260909000,holdout_seed_count=100,opponents=NAMES[:7],both_seats=True,target_overall=.9,target_each=.9,locked_before_first_holdout=True)
write(R/'RELEASE_FREEZE.json',freeze)
run(p,'release_holdout100',cfg,count=100,start=260909000)
run(p,'baseline_j7_holdout100',{},count=100,start=260909000,arm='j7')
# Same source/config, seven fresh live opponents, one holdout seed, full actions.
run(p,'release_official14',cfg,count=1,start=260909000,trace=True)
subprocess.run(['python',str(R/'tests/official_parity.py'),'release_official14','--out','tests/results/parity_release','--limit','14'],check=True)
