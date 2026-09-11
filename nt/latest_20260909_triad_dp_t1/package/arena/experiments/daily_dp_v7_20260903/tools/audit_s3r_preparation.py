"""Offline attribution. Does not run new matches or expose rival private state."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    src=Path(a.audit);out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    receipt=json.loads((src/'summary.json').read_text());assert receipt['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    result=[];cache={};hashes={}
    for s in receipt['summary']:
        path=src/f"{s['label']}_{s['opponent']}.json.gz"
        with gzip.open(path,'rt') as f:rows=json.load(f)['rows']
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        cache[s['label'],s['opponent']]={(r['seed'],r['seat']):r for r in rows}
        daily=[]
        for day in range(30):
            count=[]
            for r in rows:
                current=r['planning'][day]['preparation_actions']
                previous=r['planning'][day-1]['preparation_actions'] if day else 0
                count.append(current-previous)
            daily.append(st.fmean(count))
        assert min(daily)>=0
        production=s['own_production'];cash=s['own_cash']
        e=dict(label=s['label'],opponent=s['opponent'],games=len(rows),cash=cash['cash'],rival_cash=s['rival_cash']['cash'],
               wins=s['wins'],wages=cash['hired'],preparation_actions=sum(daily),preparation_by_day=daily,
               preparation_active_games=sum(r['planning'][-1]['preparation_actions']>0 for r in rows),
               planted=production['planted'],placed=production['placed'],no_effect=production['no_effect'],
               environment_loss=production['environment_loss'],eod_loss=production['eod_loss'],
               unassigned=s['planner_mean_unassigned_tasks'])
        result.append(e)
    examples=[]
    for opponent in sorted({k[1] for k in cache}):
        for label,base in [('prep_only','old'),('bundle_prep','bundle_only')]:
            newer=cache[label,opponent];older=cache[base,opponent];comparisons=[]
            for key,n in newer.items():
                b=older[key];seat=n['seat']
                delta=(n['money'][seat]-n['money'][1-seat])-(b['money'][seat]-b['money'][1-seat])
                comparisons.append((delta,key))
            comparisons.sort()
            for delta,key in (comparisons[0],comparisons[-1]):
                n=newer[key];b=older[key]
                examples.append(dict(opponent=opponent,variant=label,baseline=base,seed=key[0],seat=key[1],
                    margin_change=delta,baseline_money=b['money'],variant_money=n['money'],
                    baseline_planning=b['planning'],variant_planning=n['planning'],
                    caveat='Selected worst/best development example, not independent confirmation or sole causal explanation.'))
    answer=dict(status='COMPLETE_READ_ONLY_ATTRIBUTION',build=receipt['build'],summary=result,input_hashes=hashes,
        examples=examples,extra_independent_matches=0,
        caveat='The switch is isolated; changes in wages/production later are downstream consequences, not independent interventions.')
    (out/'summary.json').write_text(json.dumps(answer,indent=2))
    lines=['# 采购期并行准备：兑现诊断','','| 配置 | 对手 | 平均提前动作 | 生效局数 | 现金 | 工资 | 未排入工作 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in result:lines.append(f"| {r['label']} | {r['opponent']} | {r['preparation_actions']:.2f} | {r['preparation_active_games']}/{r['games']} | {r['cash']:,.0f} | {r['wages']:,.0f} | {r['unassigned']:.2f} |")
    lines+=['','这是A组原对局的只读归因，不增加独立样本。动作提前不保证长期收入增加。']
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))

if __name__=='__main__':main()
