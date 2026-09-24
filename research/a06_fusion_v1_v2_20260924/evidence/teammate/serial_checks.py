"""Replay observed slow cases in isolation; do not infer sandbox performance."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--opponent',required=True);args=parser.parse_args()
    rows=[json.loads(s)for s in(HERE/'runs'/f'run_{args.opponent}'/'games.jsonl').read_text().splitlines()if s]
    assert len(rows)==400 and all(not r['error'] and r['steps']==719 for r in rows)
    chosen=[max((r for r in rows if r['candidate']==c),key=lambda r:r['max_own_noninitial_s'])for c in('V1','V2')]
    chosen.append(max(rows,key=lambda r:r['max_opponent_noninitial_s']))
    unique={(r['candidate'],r['seed'],r['seat']):r for r in chosen}
    out=[]
    for index,r in enumerate(unique.values()):
        name=f'serial_{index}_{args.opponent}'
        subprocess.run([sys.executable,str(HERE/'run.py'),'serial','--opponent',args.opponent,
            '--candidate',r['candidate'],'--seed',str(r['seed']),'--seat',str(r['seat']),
            '--workers','1','--out',name],check=True)
        row=json.loads((HERE/'runs'/name/'games.jsonl').read_text().strip())
        fields=('own_cash','opponent_cash','win','tie','steps','own_actions_sha256','opponent_actions_sha256')
        exact=all(row[f]==r[f]for f in fields)
        assert exact
        out.append(dict(candidate=r['candidate'],seed=r['seed'],seat=r['seat'],exact=exact,
            parallel_own_s=r['max_own_noninitial_s'],serial_own_s=row['max_own_noninitial_s'],
            parallel_opponent_s=r['max_opponent_noninitial_s'],serial_opponent_s=row['max_opponent_noninitial_s']))
    (HERE/f'SERIAL_{args.opponent}.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
