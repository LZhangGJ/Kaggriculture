"""Append confirmation evidence without replacing the released agent."""
from pathlib import Path
import json
import run_panel as panel
from validate_startup_supply import BINS
from validate_crop_clock import paired
HERE=Path(__file__).resolve().parent
def main():
    folder=HERE/'startup_supply_validation32';result=json.loads((folder/'RESULTS.json').read_text())
    for n,(p,h) in BINS.items():
        assert panel.sha(p)==h and result['versions'][n]['overall']['games']==704 and result['versions'][n]['overall']['errors']==0
    bestfile=HERE/'DEVELOPMENT_BEST.json';best=json.loads(bestfile.read_text())
    assert best['binary_sha256']==BINS['p16'][1]
    best['status']='BEST_DEVELOPMENT_ONLY_NEW32_NOT_ABOVE_ORIGINAL_NOT_90_PERCENT'
    best['independent_confirmation']=dict(status='COMPLETED_NO_CONFIRMED_IMPROVEMENT',results='startup_supply_validation32/RESULTS.json',seeds=[2609124000,2609124031],overall=result['versions']['p16']['overall'],paired_original=result['paired']['original_to_p16'],paired_p12=result['paired']['p12_to_p16'])
    panel.save(bestfile,best)
    rows={n:[] for n in ('original','p12')}
    for f in ('crop_clock_validation32','startup_supply_validation32'):
        for n in rows:rows[n]+=json.loads((HERE/f/n/'rows.json').read_text())
    assert all(len({(x['seed'],x['opponent'],x['opponent_seat']) for x in r})==1408 for r in rows.values())
    stats,_=paired(rows['original'],rows['p12'])
    combined=dict(seeds=64,games_per_version=1408,versions={n:panel.summarize(r) for n,r in rows.items()},paired_original_to_p12=stats,boundary='Two frozen new32 confirmation blocks; not the final100 test; no per-case confirmation tuning')
    panel.save(HERE/'P12_TWO_CONFIRMATION_BLOCKS.json',combined)
    f=HERE/'INDEPENDENT_CANDIDATE.json';prev=json.loads(f.read_text());assert prev['name']=='p12'
    prev['status']='FIRST_BLOCK_WINNER_SECOND_BLOCK_BELOW_ORIGINAL_NOT_FINAL'
    prev['additional_confirmation']=dict(source='startup_supply_validation32/RESULTS.json',overall=result['versions']['p12']['overall'],paired_original=result['paired']['original_to_p12'])
    prev['combined_two_blocks']=combined
    panel.save(f,prev)
    print(json.dumps(dict(p16=result['versions']['p16']['overall'],p12_two_blocks=combined)),flush=True)
if __name__=='__main__':main()
