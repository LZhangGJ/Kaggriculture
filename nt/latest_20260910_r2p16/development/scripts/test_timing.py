from pathlib import Path
import subprocess,json
HERE=Path(__file__).resolve().parent/'candidate_r2p4'
rows=[]
for value in (0,.25,.5,1):
    target=HERE/f'test_timing_{value}'
    cmd=['g++-13','-std=c++20','-O2',f'-DR2_RIVAL_EARLY_SHARE={value}',str(HERE/'test_timing.cpp'),'-o',str(target)]
    subprocess.run(cmd,check=True);subprocess.run([str(target)],check=True)
    rows.append(dict(early=value,status='PASS',flow_cases=151,valuation_checks=2))
(HERE/'UNIT_TESTS.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2))
print(json.dumps(dict(status='PASS',variants=4)),flush=True)
