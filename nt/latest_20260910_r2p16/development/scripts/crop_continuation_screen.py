from pathlib import Path
import json,argparse
import run_panel as panel
import economic_screen as screen
HERE=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=8);p.add_argument('--names',nargs='+',default=['rotation_only','rotation_and_repeat']);args=p.parse_args()
    assert (HERE/'fert_portfolio_screen8/fertportfolio1/fertportfolio1/RESULTS.json').exists(),'Do not overlap independent16worker panels'
    binary=HERE/'candidate_r2p12/policy/cropchain1.so';assert panel.sha(binary)=='f62c0a303133d28eb93380f52aaead97ff9e7b0ca2613668ac859b638f9adf63'
    options={'rotation_only':dict(rotation=1,repeat=0),'rotation_and_repeat':dict(rotation=1,repeat=1)}
    for n in args.names:assert n in options
    for n in args.names:
        screen.OPTIONS[n]=options[n];screen.run([n],args.seeds,HERE/f'crop_continuation_screen{args.seeds}'/n,True,binary)
if __name__=='__main__':main()
