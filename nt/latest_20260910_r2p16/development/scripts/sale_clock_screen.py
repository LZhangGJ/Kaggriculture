from pathlib import Path
import argparse
import economic_screen as screen
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--names',default='mpc,coupled');p.add_argument('--seeds',type=int,default=8);p.add_argument('--market',type=int,default=0);args=p.parse_args()
 for variant in args.names.split(','):
  name=f'clock_{variant}_market{args.market}';screen.OPTIONS[name]={}
  screen.run([name],args.seeds,HERE/f'clock_screen{args.seeds}'/name,True,HERE/f'candidate_r2p8/policy/{name}.so')
if __name__=='__main__':main()
