"""Frozen mechanism ablations; official live opponents, no future information."""
from pathlib import Path
import argparse
import economic_screen as screen

HERE = Path(__file__).resolve().parent
OPTIONS = {
    'no_mpc': {'scenario': 0},
    'horizon2': {'scenario': 2},
    'horizon3': {'scenario': 3},
    'service_margin': {'service_reconcile': 3},
    'horizon2_service_margin': {'scenario': 2, 'service_reconcile': 3},
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--names',default=','.join(OPTIONS))
    p.add_argument('--seeds',type=int,default=8)
    args=p.parse_args()
    assert 1 <= args.seeds <= 100
    for name in args.names.split(','):
        screen.OPTIONS[name]=OPTIONS[name]
        screen.run([name],args.seeds,HERE/f'horizon_screen{args.seeds}'/name,True,
                   HERE/'candidate_r2p2/policy/r2p2.so')

if __name__=='__main__':main()
