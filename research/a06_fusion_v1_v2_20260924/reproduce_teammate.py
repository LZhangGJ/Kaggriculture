"""Reproduce an exact sampled-policy game using an external original opponent."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
from verify_package import verify

HERE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-version',choices=('v1','v2'),default='v1')
    parser.add_argument('--opponent',choices=('student_v306','student_v463'),required=True)
    parser.add_argument('--opponent-root',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--seat',type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    verify()
    evidence=HERE/'evidence/teammate'
    protocol=json.loads((evidence/f'PROTOCOL_{args.opponent}.json').read_text())
    rival=args.opponent_root.resolve()
    for name,digest in protocol['files'].items():
        assert hashlib.sha256((rival/name).read_bytes()).hexdigest()==digest,name
    seed=args.seed if args.seed is not None else protocol['randomization']['seeds'][0]
    assert seed in protocol['randomization']['seeds'],'Choose a recorded environment seed'
    candidate=args.candidate_version.upper()
    with gzip.open(evidence/f'run_{args.opponent}_games.jsonl.gz','rt',encoding='utf-8')as stream:
        expected=next(row for line in stream if (row:=json.loads(line))['candidate']==candidate
            and row['seed']==seed and row['seat']==args.seat)
    from portable_match import evaluate
    result=evaluate((candidate,f'{args.candidate_version}/main.py',args.opponent,str(rival/'main.py'),
        seed,args.seat,protocol['randomization']['actor_seeds'][f'{seed}:{args.seat}']))
    fields=('own_cash','opponent_cash','win','tie','steps','own_actions_sha256','opponent_actions_sha256')
    result['recorded_result_matches']=not result['error'] and all(result.get(f)==expected[f]for f in fields)
    print(json.dumps(result,ensure_ascii=False))
    assert result['recorded_result_matches'],result
if __name__=='__main__':main()
