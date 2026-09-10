from pathlib import Path
import json
import run_panel as panel
from validate_crop_clock import BINARIES
HERE=Path(__file__).resolve().parent
def main():
    r=json.loads((HERE/'crop_clock_validation32/RESULTS.json').read_text())
    assert set(r['versions'])==set(BINARIES)
    for n,(p,h) in BINARIES.items():
        assert panel.sha(p)==h and r['versions'][n]['overall']['games']==704 and r['versions'][n]['overall']['errors']==0
    previous=json.loads((HERE/'DEVELOPMENT_BEST.json').read_text())
    if previous['name']=='R2P11 observed crop clock + P9 split sale':
        previous['status']='BEST_DEVELOPMENT_ONLY_NEW32_NOT_BEST_NOT_90_PERCENT'
        previous['independent_confirmation']=dict(status='COMPLETED_NOT_90_PERCENT',results='crop_clock_validation32/RESULTS.json',seeds=[2609123000,2609123031],overall=r['versions']['p11']['overall'],paired_original=r['paired']['original_to_p11'])
        panel.save(HERE/'DEVELOPMENT_BEST.json',previous)
    best=max(r['versions'],key=lambda k:r['versions'][k]['overall']['r2_win_rate']);path,h=BINARIES[best]
    result=dict(name=best,status='BEST_OF_FOUR_ON_FROZEN_NEW32_NOT_FINAL_VALIDATION',binary=str(path.relative_to(HERE)),binary_sha256=h,results='crop_clock_validation32/RESULTS.json',source_protocol='crop_clock_validation32/PROTOCOL.json',original_config=str((panel.old.R2/'config.json').relative_to(HERE.parents[1])),overall=r['versions'][best]['overall'],by_opponent=r['versions'][best]['by_opponent'],paired_original=r['paired'].get('original_to_'+best),seeds=[2609123000,2609123031],final_holdout_used=False,released_original_unchanged=True,boundary='Candidate selected across four policies; not a new blind final estimate. Development best remains recorded separately; do not tune against per-case confirmation labels.')
    panel.save(HERE/'INDEPENDENT_CANDIDATE.json',result);print(json.dumps(dict(name=best,overall=result['overall'])),flush=True)
if __name__=='__main__':main()
