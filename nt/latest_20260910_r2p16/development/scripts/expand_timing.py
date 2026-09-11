"""100-seed development replication of a previously frozen timing variant."""
from pathlib import Path
import argparse
import economic_screen as screen
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['quarter','quarter_half','quarter_one'],required=True);args=p.parse_args()
    name='timing_'+args.name
    screen.OPTIONS[name]={}
    screen.run([name],100,HERE/'timing_full100'/args.name,True,HERE/f'candidate_r2p4/policy/{name}.so')
