"""Bounded mechanism-combination screen; all switches already have OFF parity.

No per-opponent routing, no unseen validation-seed reuse, no oracle selection.
"""
from pathlib import Path
import argparse,json
import economic_screen as screen
HERE=Path(__file__).resolve().parent
OPTIONS={
    'wide_original':({'portfolio_passes':9,'candidate_extra':2},0),
    'sale_wide':({'portfolio_passes':9,'candidate_extra':2},1),
    'sale_service':({'service_reconcile':3},1),
    'sale_no_mpc':({'scenario':0},1),
    'sale_wide_service':({'portfolio_passes':9,'candidate_extra':2,'service_reconcile':3},1),
}
def main():
    p=argparse.ArgumentParser();p.add_argument('--names',default=','.join(OPTIONS));p.add_argument('--seeds',type=int,default=8);args=p.parse_args()
    assert (HERE/'local_sale_validation32/RESULTS.json').exists(), 'Complete pre-frozen P9 confirmation first.'
    assert 1<=args.seeds<=100
    for name in args.names.split(','):
        config,mode=OPTIONS[name];screen.OPTIONS[name]=config
        screen.run([name],args.seeds,HERE/f'local_sale_combination_screen{args.seeds}'/name,True,HERE/f'candidate_r2p9/policy/local_sale_{mode}.so')
if __name__=='__main__':main()
