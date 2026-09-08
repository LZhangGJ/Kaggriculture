"""Independent capital-time ranking and existing cashflow valuation factors."""
from pathlib import Path
import hashlib,json,shutil
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=EXP/'profiles/s3w';out.mkdir(exist_ok=False)
    base=json.loads((EXP/'profiles/s3v/configs.json').read_text())['old'];configs={}
    for label,mode,rank in [('old',0,False),('calendar',2,False),('turnover',0,True),('calendar_turnover',2,True)]:
        configs[label]=dict(base,cashflow_value_mode=mode,capital_time_rank=rank)
    path=out/'configs.json';path.write_text(json.dumps(configs,indent=2),encoding='utf8')
    source=out/'source';source.mkdir();hashes={}
    for name in ('policy.hpp','module.cpp','test_policy.cpp','investment_audit.hpp','production_audit.hpp'):
        p=EXP/'native'/name;shutil.copy2(p,source/name);hashes[p.relative_to(EXP).as_posix()]=sha(p)
    (out/'freeze.json').write_text(json.dumps(dict(config_sha256=sha(path),source_hashes=hashes,
        A=[20261401,20261450],B=[20261501,20261550],new_M=[20262601,20262650],final_holdout_used=False,
        rationale='Observed timely starts, early revenue gap; test cash-lock duration, not force cash reserve or replay schedule.'),indent=2))
    print(json.dumps(dict(status='FROZEN',config_sha256=sha(path))))
if __name__=='__main__':main()
