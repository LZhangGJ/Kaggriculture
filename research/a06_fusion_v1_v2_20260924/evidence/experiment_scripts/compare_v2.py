"""Compare completed candidate rows on the same intentionally difficult development panel."""
from pathlib import Path
import collections
import csv
import json

HERE=Path(__file__).resolve().parent


def main():
    candidates=collections.defaultdict(list)
    for name in ('development_v2','development_v2_followup','development_v2_dose','development_v2_refine'):
        path=HERE/'runs'/name/'games.jsonl'
        if path.is_file():
            for line in path.read_text().splitlines():
                r=json.loads(line);candidates[r['local']].append(r)
    results=[]
    for name,rows in candidates.items():
        keys={(r['public'],r['seed'],r['local_seat']) for r in rows};assert len(keys)==len(rows)
        inside=[r for r in rows if r['public'].startswith('internal/')]
        outside=[r for r in rows if r['public'].startswith('external/')]
        def wins(x):return sum(r['local_win'] for r in x)
        values=dict(candidate=name,games=len(rows),complete=len(rows)==160,internal_wins=wins(inside),external_wins=wins(outside),
                    internal_games=len(inside),external_games=len(outside),errors=sum(bool(r['error']) for r in rows),
                    mean_margin=sum(r['margin'] for r in rows)/len(rows),
                    per_opponent={o:dict(wins=wins([r for r in rows if r['public']==o]),games=sum(r['public']==o for r in rows)) for o in sorted({r['public'] for r in rows})})
        results.append(values)
    results.sort(key=lambda x:(x['complete'],x['external_wins'],x['internal_wins'],x['mean_margin']),reverse=True)
    (HERE/'V2_DEVELOPMENT_COMPARISON.json').write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(r['candidate'],f"{r['games']}/160",f"internal {r['internal_wins']}/{r['internal_games']}",f"external {r['external_wins']}/{r['external_games']}")


if __name__=='__main__':main()
