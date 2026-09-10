from pathlib import Path
import argparse
import economic_screen as screen
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--modes',default='1,2');p.add_argument('--seeds',type=int,default=8);args=p.parse_args()
 for mode in map(int,args.modes.split(',')):
  name=f'market_{mode}';screen.OPTIONS[name]={}
  screen.run([name],args.seeds,HERE/f'market_screen{args.seeds}'/name,True,HERE/f'candidate_r2p7/policy/market_{mode}.so')
if __name__=='__main__':main()
