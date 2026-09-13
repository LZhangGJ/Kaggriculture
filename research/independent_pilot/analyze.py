"""Paired results with seed clustering, training-seed variation, and full coverage checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import dump,sha
from .protocol import verify_freeze


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    verify_freeze(root/'inputs/FREEZE.json')
    status=json.loads((root/'pilot_test/STATUS.json').read_text());assert status['complete'] and status['invalid']==0
    manifest=json.loads((root/'pilot_test/MANIFEST.json').read_text());roster=manifest['candidates']
    rows=[json.loads(s) for s in (root/'pilot_test/rows.jsonl').read_text().splitlines()]
    expected={(c['id'],s,o,seat) for c in roster for s in manifest['seeds'] for o in manifest['opponents'] for seat in (0,1)}
    keys={(r['candidate'],r['seed'],r['opponent'],r['seat']) for r in rows}
    assert len(rows)==len(keys) and keys==expected and all(r['valid'] and r['steps']==719 for r in rows)
    rng=np.random.default_rng(981231);seeds=manifest['seeds'];samples=rng.integers(len(seeds),size=(4000,len(seeds)))
    base=np.array([np.mean([r['win'] for r in rows if r['candidate']=='control' and r['seed']==s]) for s in seeds])
    families={};candidates={}
    def stats(rs):
        return {'games':len(rs),'wins':sum(r['win'] for r in rs),'draws':sum(r['tie'] for r in rs),
          'win_rate':float(np.mean([r['win'] for r in rs])),'match_score':float(np.mean([r['match_score'] for r in rs])),
          'mean_cash':float(np.mean([r['own_cash'] for r in rs])),'mean_margin':float(np.mean([r['margin'] for r in rs])),
          'max_action_seconds':max(r['latency_max'] for r in rs)}
    for c in roster:candidates[c['id']]=stats([r for r in rows if r['candidate']==c['id']])
    for family in ('control','untrained','search','ppo'):
        ids={c['id'] for c in roster if c['family']==family};rs=[r for r in rows if r['candidate'] in ids]
        means=np.array([np.mean([r['win'] for r in rs if r['seed']==s]) for s in seeds])
        delta=means-base;ci=np.quantile(delta[samples].mean(1),[.025,.975])
        record=stats(rs);record.update(delta_vs_control=float(delta.mean()),paired_ci95=ci.tolist(),
          by_opponent={o:stats([r for r in rs if r['opponent']==o]) for o in manifest['opponents']},
          by_seat={str(s):stats([r for r in rs if r['seat']==s]) for s in (0,1)},
          training_seed_win_rates={cid:candidates[cid]['win_rate'] for cid in sorted(ids)})
        record['decision']='advance_to_broader_development' if ci[0]>0 and record['max_action_seconds']<1 else 'no_supported_advance'
        families[family]=record
    trials={}
    for path in sorted((root/'trials').glob('*/STATUS.json')):
        d=json.loads(path.read_text());assert d['complete'];trials[path.parent.name]=d
    report={'complete':True,'games':len(rows),'unique_worlds':len(seeds),'failures':0,'families':families,'candidates':candidates,'trials':trials,
      'official_comparisons':sum(r['official_comparisons'] for r in rows),
      'selection_sha256':sha(root/'SELECTED.json'),'test_rows_sha256':sha(root/'pilot_test/rows.jsonl'),
      'freeze_sha256':sha(root/'inputs/FREEZE.json'),'promotion':False,
      'limitations':['Eight test worlds and four benchmark opponents give limited coverage and power.',
       'Intervals are conditional on three trained seeds; individual seed results are also shown.',
       'E1 uses a finite five-stage target grammar with a feedback executor; CP-SAT repairs allocations, not full worker schedules.',
       'PPO and search have different learning mixtures: PPO adds current self-play.',
       'CPU pilot budgets do not establish GPU learning efficiency or hosted execution compliance.',
       'JAX backend has been audited from source, not benchmarked on this host.']}
    dump(root/'RESULTS.json',report)
    lines=['# Independent search and PPO pilot','',f'Completed {len(rows)} test games on {len(seeds)} fresh worlds; zero failed games.','',
      '| Family | Games | Wins | Draws | Win rate | Match score | Mean cash | Win-rate change vs control (95% CI) |',
      '|---|---:|---:|---:|---:|---:|---:|---|']
    for family,r in families.items():
        lo,hi=r['paired_ci95'];lines.append(f'| {family} | {r["games"]} | {r["wins"]} | {r["draws"]} | {r["win_rate"]:.2%} | {r["match_score"]:.3f} | {r["mean_cash"]:.1f} | {r["delta_vs_control"]*100:+.2f} pp [{lo*100:+.2f}, {hi*100:+.2f}] |')
    lines+=['','No champion promotion or submission. These are local pilot results.','',*['- '+s for s in report['limitations']]]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:families[k] for k in families}),flush=True)


if __name__=='__main__':main()
