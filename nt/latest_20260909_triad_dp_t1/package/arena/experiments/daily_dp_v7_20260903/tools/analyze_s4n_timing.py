"""Compare already finished paired live policies, never select suffixes."""
from pathlib import Path
from collections import Counter,defaultdict
import gzip,hashlib,json,statistics as st
EXP=Path(__file__).resolve().parents[1];root=EXP/'receipts/s4n_timing_trace_v1'
meta=json.loads((root/'acceptance.json').read_text());assert meta['status']=='PASS_FROZEN_LIVE_TRACE_NOT_STRENGTH'
refs={(x['label'],x['opponent'],x['seed'],x['seat']):x for x in meta['rows']};pairs=[];examples=[]
def load(ref):
    path=root/ref['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
    with gzip.open(path,'rt',encoding='utf8') as f:return json.load(f)
def units(action):return [action['farmer'],*action['hands']]
def tasks(data):
    result=defaultdict(list);seat=data['seat']
    for frame in data['trace']:
        o=frame['before'];farm=o['farms'][seat]
        for u,a in enumerate(units(frame['actions'][seat])):
            op=a[0]
            if op not in ('HARVEST','WATER','CARE','FEED','FERTILIZE','COLLECT_FERTILIZER'):continue
            x,y=([farm['farmer'],*farm['hands']])[u];t=farm['tiles'][y][x]
            if not isinstance(t,dict):continue
            identity=(o['day'],y*10+x,t.get('crop',t.get('animal')),t.get('planted_day',t.get('placed_day')),op)
            result[identity].append(dict(step=o['step'],unit=u,action=a,cargo=o['private']['inventories'][u],yield_units=t.get('yield_units',0)))
    return result
def brief(frame,seat):
    o=frame['before'];f=o['farms'][seat]
    return dict(step=o['step'],day=o['day'],hour=o['hour'],cash=f['money'],positions=[f['farmer'],*f['hands']],
        inventories=o['private']['inventories'],shed=o['private']['shed'],seed=o['private']['seeds'],
        actions=frame['actions'][seat],target=frame['debug']['target'],insertion=frame['insertion_after'])
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    for seed in range(20262701,20262709):
        for seat in (0,1):
            a=load(refs['all_intraday',opponent,seed,seat]);b=load(refs['all_intraday_insert',opponent,seed,seat])
            first={k:None for k in ('unit_action','market_action','cash','target','physical_field')}
            for k,(x,y) in enumerate(zip(a['trace'],b['trace'])):
                ox,oy=x['before'],y['before'];fx,fy=ox['farms'][seat],oy['farms'][seat]
                differences={'unit_action':units(x['actions'][seat])!=units(y['actions'][seat]),
                    'market_action':x['actions'][seat]['market']!=y['actions'][seat]['market'],
                    'cash':fx['money']!=fy['money'],
                    'target':sorted(x['debug']['target'])!=sorted(y['debug']['target']),
                    'physical_field':fx['tiles']!=fy['tiles']}
                for name,changed in differences.items():
                    if changed and first[name] is None:first[name]=k
            counts=Counter();byop=defaultdict(Counter);changes=[];ta,tb=tasks(a),tasks(b)
            for key in ta.keys()&tb.keys():
                # Only compare equal multiplicity of this plot/lifecycle/day.
                if len(ta[key])!=len(tb[key]):continue
                for x,y in zip(ta[key],tb[key]):
                    counts['common']+=1;delta=y['step']-x['step'];kind='earlier' if delta<0 else 'later' if delta>0 else 'same_time'
                    counts[kind]+=1;byop[key[-1]][kind]+=1
                    if x['unit']!=y['unit']:counts['worker_changed']+=1;byop[key[-1]]['worker_changed']+=1
                    if delta:changes.append(dict(key=key,old=x,new=y,delta=delta))
            r0=refs['all_intraday',opponent,seed,seat];r1=refs['all_intraday_insert',opponent,seed,seat]
            result=dict(opponent=opponent,seed=seed,seat=seat,first=first,common_tasks=dict(counts),by_op={k:dict(v) for k,v in byop.items()},
                cash_delta=r1['cash']-r0['cash'],margin_delta=r1['margin']-r0['margin'],old_win=r0['win'],new_win=r1['win'])
            pairs.append(result)
            pivots=sorted({x for x in first.values() if x is not None})
            examples.append(dict(**result,first_windows=[dict(pivot=p,old=[brief(x,seat) for x in a['trace'][max(0,p-1):min(719,p+6)]],
                new=[brief(x,seat) for x in b['trace'][max(0,p-1):min(719,p+6)]]) for p in pivots],task_changes=changes[:50]))
groups=[]
for opponent in ('g001','g003','boatlee_v29','yhay81_six_day'):
    rs=[r for r in pairs if r['opponent']==opponent];allkeys=set().union(*(r['common_tasks'] for r in rs))
    groups.append(dict(opponent=opponent,pairs=len(rs),cash_delta=st.fmean(r['cash_delta'] for r in rs),margin_delta=st.fmean(r['margin_delta'] for r in rs),
        different_unit_games=sum(r['first']['unit_action'] is not None for r in rs),different_target_games=sum(r['first']['target'] is not None for r in rs),
        mean_first={k:st.fmean(r['first'][k] for r in rs if r['first'][k] is not None) if any(r['first'][k] is not None for r in rs) else None for k in rs[0]['first']},
        mean_common_tasks={k:st.fmean(r['common_tasks'].get(k,0) for r in rs) for k in allkeys}))
out=EXP/'receipts/s4n_timing_analysis_v1';out.mkdir(exist_ok=False)
with gzip.open(out/'examples.json.gz','wt',encoding='utf8') as f:json.dump(examples,f)
(out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_REAL_TIMING_COMPARISON',groups=groups,pairs=pairs,
    trace_acceptance_sha256=hashlib.sha256((root/'acceptance.json').read_bytes()).hexdigest(),
    caveat='Same lifecycle task matching does not make downstream states equal; timing changes and cash differences are full-policy observations, not isolated marginal causal labels. N first8, not validation.'),indent=2))
print(json.dumps(groups,indent=2))
