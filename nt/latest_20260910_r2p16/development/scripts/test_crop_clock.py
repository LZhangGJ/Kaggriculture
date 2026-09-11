from pathlib import Path
import subprocess,json,hashlib,time
HERE=Path(__file__).resolve().parent/'candidate_r2p11'
cmd=['g++-13','-std=c++20','-O2','-ffp-contract=off',str(HERE/'test_crop_clock.cpp'),str(HERE/'policy/executor/vendor/simulator.cpp'),'-o',str(HERE/'test_crop_clock')]
started=time.perf_counter();subprocess.run(cmd,check=True);r=subprocess.run([str(HERE/'test_crop_clock')],check=True,text=True,capture_output=True)
result=dict(status='PASS',output=r.stdout,seconds=time.perf_counter()-started,command=cmd,
    sources={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ['test_crop_clock.cpp','policy/observed_crop_clock.hpp','policy/planner.hpp']})
(HERE/'UNIT_TESTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
