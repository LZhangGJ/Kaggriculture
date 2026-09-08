"""Freeze terminal capacity scheduling and its execution-scheduler interaction."""
from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'profiles/s3u';out.mkdir(parents=True,exist_ok=False)
    base=json.loads((EXP/'profiles/s3t/configs.json').read_text())['old'];configs={}
    for label,compile_on,terminal_on in [('old',False,False),('compile_only',True,False),('terminal_only',False,True),('compile_terminal',True,True)]:
        configs[label]=dict(base,regret_compile=compile_on,terminal_deposit_schedule=terminal_on)
    path=out/'configs.json';path.write_text(json.dumps(configs,indent=2),encoding='utf8')
    src=out/'source';src.mkdir();hashes={}
    for name in ('policy.hpp','module.cpp','test_policy.cpp'):
        original=EXP/'native'/name;shutil.copy2(original,src/name);hashes[original.relative_to(EXP).as_posix()]=sha(original)
    (out/'freeze.json').write_text(json.dumps(dict(configurations=configs,config_sha256=sha(path),source_hashes=hashes,
        A=[20261401,20261450],B=[20261501,20261550],conditional_new_K=[20262401,20262450],final_holdout_used=False,fitted_parameters=0),indent=2))
    print(json.dumps(dict(status='FROZEN',config_sha256=sha(path),source_hashes=hashes)))
if __name__=='__main__':main()
