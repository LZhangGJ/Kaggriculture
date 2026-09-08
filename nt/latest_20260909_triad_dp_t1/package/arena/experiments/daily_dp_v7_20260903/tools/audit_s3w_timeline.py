"""Read-only investment timing; no hidden future is supplied to a policy."""
from pathlib import Path
from collections import defaultdict, deque, Counter
import gzip, hashlib, json, statistics as st

EXP=Path(__file__).resolve().parents[1]
KINDS={0:'wheat',1:'carrot',2:'tomato',3:'strawberry',4:'melon',9:'goose',10:'cow',11:'sheep'}
PRODUCT={**{i:i for i in range(5)},9:5,10:6,11:7}
DELAY={0:4,1:3,2:8,3:10,4:10,9:4,10:8,11:6}
PLANT,PLACE,HARVEST,FERT,SEED,ANIMAL,SELL=8,7,10,16,20,22,23

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)
def distribution(xs):
    if not xs:return dict(n=0)
    x=sorted(xs)
    return dict(n=len(x),mean=st.fmean(x),p50=st.median(x),p90=x[min(len(x)-1,int(.9*len(x)))],maximum=max(x),zero=sum(v==0 for v in x))

def main():
    src=EXP/'receipts/s3w_investment_A50_v1';out=EXP/'receipts/s3w_timeline_summary_v1';out.mkdir(exist_ok=False)
    meta=json.loads((src/'summary.json').read_text());assert meta['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    results=[];hashes={};checked=0;witnesses=[]
    for entry in meta['summary']:
        opponent=entry['opponent'];p=src/f'old_{opponent}.json.gz';hashes[p.name]=sha(p)
        rows=load(p)['rows'];oldpath=EXP/f'receipts/s3v_idle_audit_A50_v1/old_{opponent}.json.gz'
        hashes['prior/'+oldpath.name]=sha(oldpath);old={(r['seed'],r['seat']):r for r in load(oldpath)['rows']}
        stats={k:dict(ages=[],harvest_lag=[],start_days=[],proposal=Counter(),first_sales=[],unstarted_seeds=[]) for k in KINDS}
        totals=Counter();phase=[]
        for r in rows:
            key=(r['seed'],r['seat']);previous=old[key]
            for name,value in previous.items():assert r[name]==value,('changed offline audit',opponent,key,name)
            checked+=1;seat=r['seat'];inv=r['investments'];cohorts=inv['cohorts'];events=inv['events']
            by_kind=Counter(c['kind'] for c in cohorts);acquired=Counter();bought=Counter();sold=Counter()
            for step,op,k,q,u,pos in events:
                if op in (HARVEST,FERT):acquired[k]+=q
                if op==SEED:bought[k]+=q
                if op==ANIMAL:bought[k]+=q
                if op==SELL:sold[k]+=q
            for k in KINDS:
                assert by_kind[k]==sum(d[seat]['planted'][k] if k<5 else d[seat]['placed'][k-9] for d in r['production'])
                assert bought[k]==sum(d[seat]['seed_bought'][k] if k<5 else d[seat]['bought'][k] for d in r['production'])
            for k in range(9):
                assert acquired[k]==sum(d[seat]['acquired'][k] for d in r['production'])
                assert sold[k]==sum(d[seat]['sold'][k] for d in r['production'])
            # Seeds are fungible and non-perishable. FIFO is an age-accounting
            # convention, not an inference about intention or causation.
            fifo=defaultdict(deque)
            for step,op,k,q,u,pos in events:
                if op in (SEED,ANIMAL):fifo[k].extend([step]*q)
                elif op in (PLANT,PLACE):
                    for _ in range(q):
                        assert fifo[k],('unfunded actual start',key,step,k)
                        when=fifo[k].popleft();assert step>=when
                        stats[k]['ages'].append(step//24-when//24)
            for k in KINDS:
                remaining=r['production'][-1][seat]['end_seeds'][k] if k<5 else r['production'][-1][seat]['end_private'][k]
                # This pool has no animal inventory discard; if it ever does,
                # do not silently assign those animals to a later placement.
                assert len(fifo[k])==remaining,('FIFO inventory loss needs explicit handling',key,k)
                stats[k]['unstarted_seeds'].append(remaining)
            starts=defaultdict(list)
            for c in cohorts:
                k=c['kind'];starts[c['position'],k].append(c['start_step']);stats[k]['start_days'].append(c['start_step']//24)
                if c['first_harvest_step']>=0:
                    lag=c['first_harvest_step']//24-c['start_step']//24-DELAY[k];stats[k]['harvest_lag'].append(lag)
                else:totals['cohorts_without_main_harvest']+=1
                totals['cohorts']+=1
            for p in inv['proposals']:
                k=p['kind'];later=[t for t in starts[p['position'],k] if t>=p['step']]
                if not later:category='no_later_same_slot_start'
                else:
                    delta=min(later)//24-p['step']//24
                    category='same_day' if delta==0 else 'next_day' if delta==1 else 'later'
                stats[k]['proposal'][category]+=1
            for k in KINDS:
                # Main product only; excludes earlier financing from fertilizer.
                product=PRODUCT[k];hs=[t for t,op,i,*_ in events if op==HARVEST and i==product]
                ss=[t for t,op,i,*_ in events if op==SELL and i==product]
                if hs and ss:stats[k]['first_sales'].append(min(ss)//24-min(hs)//24)
            for d in (0,3,7,11,15,19,23,29):
                own=[x[seat] for x in r['cash_ledger'][:d+1]];rival=[x[1-seat] for x in r['cash_ledger'][:d+1]]
                phase.append(dict(day=d,own_cash=own[-1]['end'],rival_cash=rival[-1]['end'],
                    own_income=sum(sum(x['sales']) for x in own),rival_income=sum(sum(x['sales']) for x in rival),
                    own_invest=sum(sum(x['seeds'])+sum(x['animals'])+x['land'] for x in own),
                    rival_invest=sum(sum(x['seeds'])+sum(x['animals'])+x['land'] for x in rival)))
        kinds=[]
        for k,x in stats.items():kinds.append(dict(kind=KINDS[k],purchase_to_start_days=distribution(x['ages']),
            first_harvest_minus_nominal_days=distribution(x['harvest_lag']),start_day=distribution(x['start_days']),
            proposal=dict(x['proposal']),first_product_harvest_to_first_sale_days=distribution(x['first_sales']),
            remaining_unstarted_mean=st.fmean(x['unstarted_seeds'])))
        phases=[]
        for d in sorted({x['day'] for x in phase}):
            group=[x for x in phase if x['day']==d];phases.append({k:st.fmean(x[k] for x in group) for k in group[0]})
        results.append(dict(opponent=opponent,games=len(rows),kinds=kinds,phases=phases,totals=dict(totals)))
    report=dict(status='COMPLETE_OBSERVATIONAL_TIMELINE_NOT_CAUSAL',unchanged_audit_games=checked,
        no_policy_or_rules_or_opponent_change=True,build=meta['build'],source_hashes=hashes,results=results,
        caveats=['Repeated diagnostics are not new independent games.', 'FIFO is a fungible input age convention.',
                 'Later same tile/kind is not proof of a persistent commitment.',
                 'First main-product sale excludes animal fertilizer and is not project break-even.',
                 'Collection lag can be intentional batching; low cash can be productive investment.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for r in results:
        print(r['opponent'],[(k['kind'],round(k['purchase_to_start_days'].get('mean',0),2),round(k['first_harvest_minus_nominal_days'].get('mean',0),2),k['proposal']) for k in r['kinds']],flush=True)
    print(json.dumps(dict(status=report['status'],unchanged_audit_games=checked)))
if __name__=='__main__':main()
