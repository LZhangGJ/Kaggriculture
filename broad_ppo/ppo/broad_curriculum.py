"""Bounded, deterministic curriculum. No model I/O or global RNG use here."""
import copy
import hashlib
import math

SCHEMA = 'broad-ppo-v1'

def validate_registry(registry):
    entries=registry['entries']; seen=set(); pools={}
    for r in entries:
        key=r['sha256']
        if key in seen or len(key)!=64: raise ValueError('Duplicate/invalid registry identity')
        seen.add(key)
        if r['pool'] not in ('practice','development','audit'): raise ValueError('Invalid pool')
        pools.setdefault(r['family'],set()).add(r['pool'])
        if r['pool']=='audit' and r.get('bc_exposure')!='verified_absent':
            raise ValueError('Unverified BC exposure cannot be held out')
        if r.get('proxy') and r.get('eligible') and not r.get('qualification',{}).get('passed'):
            raise ValueError('Unqualified proxy')
    if any('audit' in p and len(p)>1 for p in pools.values()): raise ValueError('Audit family leaked')
    return registry

def choose(manifest,rotation,matchups):
    retention=manifest['retention']
    block=rotation//2
    keeper=retention[block%len(retention)]
    pool=[r for r in manifest['roster'] if r['sha256'] not in {x['sha256'] for x in retention}]
    if not pool: pool=retention
    families={}
    for r in pool: families.setdefault(r['family'],[]).append(r)
    names=sorted(families)
    difficulty=[]
    for name in names:
        scores=[]
        for row in families[name]:
            stat=matchups.get(row['sha256'],{})
            # Shrink uncertain results and keep a floor even for mastered families.
            score=(stat.get('effective_score_sum',0.)+16)/(stat.get('effective_games',0)+32)
            scores.append(max(.1,1-score))
        difficulty.append(sum(scores)/len(scores))
    weights=[.5/len(names)+.5*d/sum(difficulty) for d in difficulty]
    key=f"{manifest['selection_seed']}:{block}".encode()
    value=int(hashlib.sha256(key).hexdigest()[:16],16)/2**64
    cumulative=0.; selected=names[-1]
    for name,w in zip(names,weights):
        cumulative+=w
        if value<cumulative: selected=name;break
    rows=sorted(families[selected],key=lambda x:x['sha256'])
    opponent=rows[int(hashlib.sha256(key+b':member').hexdigest()[:16],16)%len(rows)]
    return keeper,opponent,dict(block=block,family=selected,weights=dict(zip(names,weights)))

def rebuild(value):
    value=copy.deepcopy(value);unique={}
    for r in value['retention'][:2]+value.get('recent',[])[:2]+value.get('coverage',[])[:4]:
        unique.setdefault(r['sha256'],r)
    value['roster']=list(unique.values());value['version']+=1
    approved={r['sha256']:r for r in [*value['roster'],value['champion'],value['teacher']]}
    value['approved']=approved
    return value

def admit(value,entry):
    value=copy.deepcopy(value)
    entry=dict(entry,family='ppo-lineage',pool='practice',decoding='sampled',eligible=True)
    value['recent']=([entry]+[r for r in value.get('recent',[]) if r['sha256']!=entry['sha256']])[:2]
    return rebuild(value)

def family_scores(summary,panel):
    groups={}
    for row in panel['opponents']:
        s=summary['opponents'][row['name']]
        if s['failures'] or s['valid']!=s['games']:raise ValueError('Invalid evaluation')
        groups.setdefault(row['family'],[]).append(s['score'])
    values=sorted(sum(x)/len(x) for x in groups.values())
    return dict(mean=sum(values)/len(values),lower_quartile=sum(values[:max(1,math.ceil(len(values)/4))])/max(1,math.ceil(len(values)/4)),families={k:sum(v)/len(v) for k,v in groups.items()})

def paired_comparison(a,b,panel):
    import numpy as np
    def indexed(rows):
        out={}
        for r in rows:
            key=(r['opponent'],r['seed'],r['seat'])
            if key in out or not(r.get('valid') and r.get('terminal') and r.get('resolved')):raise ValueError('Invalid/duplicate result')
            out[key]=r['score']
        return out
    a,b=indexed(a),indexed(b)
    expected={(o['name'],s,seat) for o in panel['opponents'] for s in o['seeds'] for seat in (0,1)}
    if set(a)!=expected or set(b)!=expected:raise ValueError('Incomplete paired evaluation')
    seeds=panel['opponents'][0]['seeds']
    if any(o['seeds']!=seeds for o in panel['opponents']):raise ValueError('Whole seed clusters required')
    families=sorted({o['family'] for o in panel['opponents']})
    deltas=[]
    for seed in seeds:
        fs=[]
        for family in families:
            names=[o['name'] for o in panel['opponents'] if o['family']==family]
            fs.append(sum(a[n,seed,s]-b[n,seed,s] for n in names for s in (0,1))/(2*len(names)))
        deltas.append(sum(fs)/len(fs))
    x=np.array(deltas);rng=np.random.default_rng(620260920)
    samples=x[rng.integers(0,len(x),(20000,len(x)))].mean(1)
    ci=np.quantile(samples,[.025,.975]).tolist()
    return dict(delta=float(x.mean()),ci95=ci,accepted=ci[0]>0,seed_clusters=len(x),bootstrap=20000)
