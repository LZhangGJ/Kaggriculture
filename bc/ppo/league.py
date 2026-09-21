"""Small frozen league. Admission is explicit; periodic saves are not promotions."""
import hashlib
import json
from pathlib import Path
import numpy as np
import copy


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def load_league(path):
    data = json.loads(Path(path).read_text())
    rows = data['opponents']
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Duplicate opponent IDs')
    if any(r['id'] in ('learner','bc') or r['id'].startswith('script:') for r in rows):
        raise ValueError('Reserved policy ID')
    if any(sum(r['kind']==kind for r in rows)!=1 for kind in ('self','bc')):
        raise ValueError('League requires exactly one self and one BC row')
    if any(r['weight']<0 for r in rows) or abs(sum(r['weight'] for r in rows)-1)>1e-8:
        raise ValueError('League weights must sum to one')
    for r in rows:
        if r['kind'] not in ('self','bc','checkpoint','script'): raise ValueError('Unknown opponent kind')
        if r['kind'] in ('checkpoint','script') and r.get('path') not in ('starter','random','pass'):
            source = Path(r['path']).resolve()
            if sha(source) != r['sha256']: raise ValueError('Opponent SHA mismatch: '+r['id'])
            r['path'] = str(source)
        if r['kind']=='script' and r['path']=='pass': raise ValueError('PASS is evaluation-only')
    return data


def assignments(league, seed, update, games, world, rank, balanced=False):
    """Build globally, then shard. Each seed and game has one owner."""
    rng = np.random.default_rng(seed+update)
    rows = league['opponents']
    counts = np.floor(np.array([r['weight'] for r in rows])*games).astype(int)
    residual = np.array([r['weight'] for r in rows])*games-counts
    for i in np.argsort(-residual)[:games-int(counts.sum())]: counts[i]+=1
    choices = np.repeat(np.arange(len(rows)),counts)
    rng.shuffle(choices)
    offsets=np.cumsum(counts)-counts
    seen=np.zeros(len(rows),dtype=int)
    out = []
    for i,choice in enumerate(choices):
        r = rows[choice]
        seat = (i+update)%2
        policy = 'learner' if r['kind']=='self' else 'bc' if r['kind']=='bc' else ('script:'+r['path']) if r['kind']=='script' else r['id']
        policies = [policy,policy]
        policies[seat] = 'learner'
        # Seeds advance monotonically across updates; check eval disjointness before collection.
        a = dict(game=update*games+i,seed=seed+update*games+i,policies=policies,
                 family=r['id'],learner_seats=[s for s,p in enumerate(policies) if p=='learner'])
        # Keep per-policy GPU batch sizes stable across shuffles. Consecutive
        # family blocks round-robin evenly even when a family has a remainder.
        owner=int((offsets[choice]+seen[choice])%world) if balanced else i%world
        seen[choice]+=1
        if owner==rank: out.append(a)
    return out


def adapt_history(league, results):
    """Keep history's total share fixed; half uniform, half difficulty weighted."""
    value=copy.deepcopy(league)
    rows=[r for r in value['opponents'] if r['kind']=='checkpoint']
    if rows:
        mass=sum(r['weight'] for r in rows)
        hard=np.array([max(.1,1-results.get(r['id'],.5)) for r in rows])
        weights=.5/len(rows)+.5*hard/hard.sum()
        for row,w in zip(rows,weights): row['weight']=float(mass*w)
    return value


def update_matchups(results, games):
    value=dict(results)
    groups={}
    for game in games:
        if any(game['faults']) or len(game['learner_seats'])!=1: continue
        seat=game['learner_seats'][0];cash=game['cash']
        score=.5 if cash[seat]==cash[1-seat] else float(cash[seat]>cash[1-seat])
        groups.setdefault(game['family'],[]).append(score)
    for key,rows in groups.items(): value[key]=.8*value.get(key,.5)+.2*float(np.mean(rows))
    return value


def add_snapshot(path, checkpoint, identifier, family, replace=None, history_mass=.25):
    data = load_league(path)
    if identifier in ('learner','bc') or identifier.startswith('script:'): raise ValueError('Reserved policy ID')
    if any(r['id']==identifier for r in data['opponents']): raise ValueError('Snapshot ID exists')
    if replace:
        old = next(r for r in data['opponents'] if r['id']==replace)
        if old['kind']!='checkpoint': raise ValueError('Only historical snapshots may be replaced')
        data['opponents'].remove(old)
    else:
        if sum(r['kind']=='checkpoint' for r in data['opponents'])>=8: raise ValueError('History is full; name a replacement')
    data['opponents'].append(dict(id=identifier,kind='checkpoint',path=str(Path(checkpoint).resolve()),
                                 sha256=sha(checkpoint),weight=0.,family=family))
    history=[r for r in data['opponents'] if r['kind']=='checkpoint']
    fixed=sum(r['weight'] for r in data['opponents'] if r['kind'] not in ('self','checkpoint'))
    if not 0<history_mass<1-fixed: raise ValueError('History allocation leaves no self-play share')
    for row in history: row['weight']=history_mass/len(history)
    next(r for r in data['opponents'] if r['kind']=='self')['weight']=1-fixed-history_mass
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(path)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--league',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--id',required=True);p.add_argument('--family',required=True);p.add_argument('--replace')
    p.add_argument('--history-mass',type=float,default=.25)
    a=p.parse_args();add_snapshot(a.league,a.checkpoint,a.id,a.family,a.replace,a.history_mass)
