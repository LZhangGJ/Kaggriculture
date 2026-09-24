"""Read-only progress/results table; incomplete runs remain labelled partial."""
from pathlib import Path
import argparse
import collections
import json

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    out=HERE/'runs'/a.run;protocol=json.loads((out/'PROTOCOL.json').read_text())
    rows=[json.loads(s) for s in (out/'games.jsonl').read_text().splitlines() if s]
    result={}
    for ident in protocol['candidates']:
        groups={}
        for group in ('internal','external'):
            selected=[r for r in rows if r['local']==ident and r['public'].startswith(group+'/')]
            valid=[r for r in selected if r['error'] is None]
            n=len(valid)
            groups[group]=dict(games=n,wins=sum(r['local_win'] for r in valid),ties=sum(r['tie'] for r in valid),
                win_rate=sum(r['local_win'] for r in valid)/n if n else None,
                mean_margin=sum(r['margin'] for r in valid)/n if n else None,errors=len(selected)-n)
        result[ident]=groups
    status='COMPLETE' if len(rows)==protocol['total_games'] else 'PARTIAL'
    print(status,len(rows),'/',protocol['total_games'])
    for ident,g in sorted(result.items(),key=lambda item:min(item[1][x]['win_rate'] or 0 for x in ('internal','external')),reverse=True):
        print(ident,' '.join(f"{x}:{s['wins']}/{s['games']} ties={s['ties']} margin={round(s['mean_margin'],1) if s['mean_margin'] is not None else None}" for x,s in g.items()))
    data=dict(status=status,completed=len(rows),expected=protocol['total_games'],candidates=result)
    (out/'RESULTS.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
