"""Recompute paired win conversions from the shipped actual panel records."""
from pathlib import Path
import json,statistics,argparse
R=Path(__file__).resolve().parents[1]
def key(r):return r['opponent'],r['seed'],r['opponent_seat']
def compare(a,b):
 if len(a)!=len(b)or {key(r)for r in a}!={key(r)for r in b}:raise ValueError('Unpaired records')
 other={key(r):r for r in b}
 if any(r['runtime_error']or r['steps']!=719 for r in a+b):raise ValueError('Error or incomplete game')
 return {'games':len(a),'parent_wins':sum(r['win']for r in a),'joint_wins':sum(r['win']for r in b),'rescued':sum(not r['win']and other[key(r)]['win']for r in a),'lost_old_wins':sum(r['win']and not other[key(r)]['win']for r in a),'mean_margin_delta':statistics.mean(other[key(r)]['margin']-r['margin']for r in a)}
def main():
 def load(name):return json.loads((R/'evidence/panels'/name/'rows.json').read_text())
 a,b,c,d=map(load,['base3','joint_v2a_3','base5','joint5'])
 out={'development':compare(a,b),'confirmation':compare(c,d),'overall':compare(a+c,b+d)}
 saved=json.loads((R/'evidence/RESULTS.json').read_text())
 for k in ['development','confirmation','overall']:
  expected=saved['overall']if k=='overall'else saved['by_block'][k]
  for f in ['games','parent_wins','joint_wins','rescued','lost_old_wins']:
   if expected[f]!=out[k][f]:raise ValueError(f'Published aggregate mismatch {k}/{f}')
 print(json.dumps({'status':'PASS_RECALCULATED_FROM_ROWS','results':out},indent=2))
if __name__=='__main__':main()
