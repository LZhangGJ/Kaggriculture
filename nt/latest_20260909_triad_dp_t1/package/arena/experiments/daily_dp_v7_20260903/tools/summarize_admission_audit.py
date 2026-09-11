"""Compare funded work with hired capacity; diagnostic bounds, not wins."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.audit);src=root/'summary.json';receipt=json.loads(src.read_text())
    assert receipt['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    out=Path(a.out);out.mkdir(exist_ok=False,parents=True);result=[]
    for entry in receipt['summary']:
        path=root/f"{entry['label']}_{entry['opponent']}.json.gz"
        with gzip.open(path,'rt') as f:rows=json.load(f)['rows']
        days=[d for r in rows for d in r['admissions'] if d['captured']]
        assert len(days)==30*len(rows),('missing compilation observation',entry['label'],entry['opponent'])
        def group(ds):
            return dict(days=len(ds),mean_hands=statistics.fmean(d['hands'] for d in ds),
                any_resource_shortage_days=sum(any(d[k]>0 for k in ('seed_shortage','feed_shortage','fertilizer_shortage','animal_shortage')) for d in ds),
                dropped_job_days=sum(d['resource_dropped_jobs']>0 for d in ds),
                seed_shortage=sum(d['seed_shortage'] for d in ds),feed_shortage=sum(d['feed_shortage'] for d in ds),
                fertilizer_shortage=sum(d['fertilizer_shortage'] for d in ds),animal_shortage=sum(d['animal_shortage'] for d in ds),
                fewer_hands_feasible_days=sum(0<=d['min_hands']<d['hands'] for d in ds),
                excess_hands_total=sum(d['hands']-d['min_hands'] for d in ds if d['min_hands']>=0),
                conditional_saving=sum(d['conditional_wage_saving'] for d in ds),
                low_cash_lt10_days=sum(d['cash']<10 for d in ds),mean_cash=statistics.fmean(d['cash'] for d in ds))
        # Seeds do not consume shed capacity. 100 is the maximum official seed
        # price, so this test conservatively proves sufficient current cash.
        # It does not prove waiting one more turn is economically optimal.
        full=group(days);full['affordable_seed_shortage_days']=sum(d['seed_shortage']>0 and d['cash']>=100*d['seed_shortage'] for d in days)
        full['affordable_missing_seed_units']=sum(d['seed_shortage'] for d in days if d['seed_shortage']>0 and d['cash']>=100*d['seed_shortage'])
        stage=[dict(days_zero_based=[lo,hi],**group([d for d in days if lo<=d['day']<=hi])) for lo,hi in ((0,5),(6,11),(12,17),(18,23),(24,29))]
        broken=[]
        for r in rows:
            for d in r['admissions']:
                if d['resource_dropped_jobs']>0 and 0<=d['min_hands']<d['hands']:
                    broken.append(dict(seed=r['seed'],seat=r['seat'],**d))
        item=dict(label=entry['label'],opponent=entry['opponent'],games=len(rows),summary=full,stages=stage,
            mean_conditional_wage_saving=full['conditional_saving']/len(rows),mean_actual_wages=entry['own_cash']['hired'],
            shortage_and_overstaff_examples=sorted(broken,key=lambda d:-d['conditional_wage_saving'])[:10],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        result.append(item)
    data=dict(status='COMPLETE_READ_ONLY_DIAGNOSTIC_NOT_POLICY_ACCEPTANCE',results=result,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        caveat='Fewer-hands feasibility is only a conditional upper bound on wage savings. Altering hires may change finance, routing, sale time and opponent decisions. No causal gain is asserted.')
    (out/'summary.json').write_text(json.dumps(data,indent=2))
    lines=['# 采购可执行性和雇工诊断','','每对手100局；各30个日初编排窗口。人数下界仅指当前算法找到的可行更少人数，不是数学最优。',
        '', '| 对手 | 缺料天数/局 | 因资源删除任务的天数/局 | 可用更少人数的天数/局 | 条件节省工资/局 | 实际工资/局 |','|---|---:|---:|---:|---:|---:|']
    for x in result:
        d=x['summary'];n=x['games'];lines.append(f"| {x['opponent']} | {d['any_resource_shortage_days']/n:.2f} | {d['dropped_job_days']/n:.2f} | {d['fewer_hands_feasible_days']/n:.2f} | {x['mean_conditional_wage_saving']:.2f} | {x['mean_actual_wages']:.2f} |")
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines))

if __name__=='__main__':main()
