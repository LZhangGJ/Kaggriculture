"""Freeze ordinary-day idle transport, independently of the joint scheduler."""
from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'profiles/s3v';out.mkdir(parents=True,exist_ok=False)
    base=json.loads((EXP/'profiles/s3u/configs.json').read_text())['terminal_only'];configs={}
    for label,compile_on,idle in [('old',False,False),('compile_only',True,False),('idle_only',False,True),('compile_idle',True,True)]:
        configs[label]=dict(base,regret_compile=compile_on,overflow_idle_dispatch=idle)
    path=out/'configs.json';path.write_text(json.dumps(configs,indent=2),encoding='utf8')
    src=out/'source';src.mkdir();hashes={}
    for name in ('policy.hpp','module.cpp','test_policy.cpp'):
        original=EXP/'native'/name;shutil.copy2(original,src/name);hashes[original.relative_to(EXP).as_posix()]=sha(original)
    (out/'freeze.json').write_text(json.dumps(dict(configurations=configs,config_sha256=sha(path),source_hashes=hashes,
        A=[20261401,20261450],B=[20261501,20261550],conditional_new_L=[20262501,20262550],final_holdout_used=False,fitted_parameters=0),indent=2))
    print(json.dumps(dict(status='FROZEN',config_sha256=sha(path),source_hashes=hashes)))
if __name__=='__main__':main()
