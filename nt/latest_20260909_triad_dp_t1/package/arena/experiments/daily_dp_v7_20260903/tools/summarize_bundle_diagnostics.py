"""Read-only attribution of the funded-bundle ablation, not extra matches."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    src=Path(a.audit);out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    receipt=json.loads((src/'summary.json').read_text());assert receipt['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    results=[]
    for s in receipt['summary']:
        path=src/f"{s['label']}_{s['opponent']}.json.gz"
        with gzip.open(path,'rt') as f:rows=json.load(f)['rows']
        assert all(len(r['planning'])==30 for r in rows)
        ended=[r['planning'][-1] for r in rows]
        e=dict(label=s['label'],opponent=s['opponent'],games=len(rows),
            evaluations=st.fmean(r['bundle_evaluations'] for r in ended),
            switches=st.fmean(r['bundle_switches'] for r in ended),
            deferred_targets=st.fmean(r['bundle_removed_targets'] for r in ended),
            changed_games=sum(r['bundle_switches']>0 for r in ended),
            resource_gap_days=st.fmean(sum(x['captured'] and x['resource_dropped_jobs']>0 for x in r['admissions']) for r in rows),
            planned_jobs_lost_for_material=st.fmean(sum(x['resource_dropped_jobs'] for x in r['admissions'] if x['captured']) for r in rows),
            cash=s['own_cash']['cash'],rival_cash=s['rival_cash']['cash'],wages=s['own_cash']['hired'],
            planted=s['own_production']['planted'],placed=s['own_production']['placed'],
            field_decay=s['own_production']['environment_loss'],eod_loss=s['own_production']['eod_loss'],
            no_effect=s['own_production']['no_effect'],unassigned=s['planner_mean_unassigned_tasks'],
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        results.append(e)
    result=dict(status='COMPLETE_READ_ONLY_DIAGNOSTIC',build=receipt['build'],results=results,
        caveat='Fewer failed tasks after proposing fewer tasks is not an economic success. Use full-game cash and wins. No extra independent scenarios.')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# 投资包兑现诊断（只读）','','| 配置 | 对手 | 每局比较数 | 每局更改天数 | 本日撤回/延期目标次数 | 缺料天数 | 我方现金 | 工资 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['label']} | {r['opponent']} | {r['evaluations']:.1f} | {r['switches']:.2f} | {r['deferred_targets']:.1f} | {r['resource_gap_days']:.2f} | {r['cash']:,.0f} | {r['wages']:,.0f} |")
    lines+=['','这是同批对局的重复审计，不是额外独立强度样本。减少目标后少缺料，并不等于更赚钱；本日延期不保证明日恢复。']
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))

if __name__=='__main__':main()
