from pathlib import Path
import subprocess,json,hashlib,time
HERE=Path(__file__).resolve().parent/'candidate_r2p14'
cmd=['g++-13','-std=c++20','-O2','-ffp-contract=off',str(HERE/'test_fert_portfolio.cpp'),str(HERE/'policy/executor/vendor/simulator.cpp'),'-o',str(HERE/'test_fert_portfolio')]
started=time.perf_counter();subprocess.run(cmd,check=True);r=subprocess.run([str(HERE/'test_fert_portfolio')],check=True,text=True,capture_output=True)
result=dict(status='PASS',output=r.stdout,seconds=time.perf_counter()-started,command=cmd,sources={p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ['test_fert_portfolio.cpp','policy/triad.hpp','policy/planner.hpp']})
(HERE/'UNIT_TESTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
