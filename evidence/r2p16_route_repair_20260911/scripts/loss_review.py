"""Locate first divergence in the largest win-to-loss changes; no policy modification."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tag')
    args = parser.parse_args()
    folder = ROOT / 'runs' / args.tag
    rows = read(folder / 'rows.json')
    pairs = defaultdict(dict)
    for row in rows:
        pairs[row['opponent'], row['seed'], row['seat']][row['arm']] = row
    report = []
    for baseline in ['workflow', 'recovery']:
        changed = [p for p in pairs.values() if p[baseline]['win'] and not p['fixed']['win']]
        for pair in sorted(changed, key=lambda p: p['fixed']['margin']-p[baseline]['margin'])[:3]:
            a, b = pair[baseline], pair['fixed']
            replays = [read(folder / 'matches' / (r['name']+'.json.gz')) for r in [a, b]]
            audits = [read(folder / 'matches' / (r['name']+'.audit.json.gz')) for r in [a, b]]
            seat = a['seat']
            first = next(i for i in range(1,720) if replays[0]['steps'][i][seat]['action'] != replays[1]['steps'][i][seat]['action'])
            step = first-1
            traces = []
            for audit in audits:
                before = [r for r in audit['route'] if r['step'] < step]
                at = [r for r in audit['route'] if r['step'] <= step]
                traces.append(dict(before=before[-1] if before else None, at=at[-1] if at else None))
            report.append(dict(baseline=baseline, opponent=a['opponent'], seed=a['seed'], seat=seat,
                first_different_step=step,
                observation_before_first_difference_equal=replays[0]['steps'][step][seat]['observation']==replays[1]['steps'][step][seat]['observation'],
                actions=[r['steps'][first][seat]['action'] for r in replays],
                timing=[audit['action_seconds'][step] for audit in audits],
                debug=traces,
                result={k:[a[k],b[k]] for k in ['cash','opponent_cash','margin','wages','hires','production','invalid_actions','uncompleted_required_tasks']},
                route_counters=[{k:r['route'].get(k,0) for k in ['budget_stops','recovery_events','recovery_joint','recovery_partial']} for r in [a,b]],
                daily=[{k:day[k] for k in ['day','cash','wages','hires','production','overflow_units']} for day in audits[1]['daily']]))
    (folder / 'LOSS_REVIEW.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([dict(baseline=r['baseline'],opponent=r['opponent'],seed=r['seed'],seat=r['seat'],first_step=r['first_different_step'],same_observation=r['observation_before_first_difference_equal'],timing=r['timing'],margin=r['result']['margin'],counters=r['route_counters']) for r in report],indent=2))


if __name__ == '__main__':
    main()
