"""Read-only shared-service diagnostics; job counts across decompositions differ."""
from pathlib import Path
import argparse,gzip,hashlib,json,statistics as st
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    src=Path(a.audit);out=Path(a.out);out.mkdir(exist_ok=False,parents=True)
    receipt=json.loads((src/'summary.json').read_text());assert receipt['status']=='PASS_OFFLINE_AUDIT_NOT_GOAL_ACCEPTANCE'
    result=[]
    for s in receipt['summary']:
        path=src/f"{s['label']}_{s['opponent']}.json.gz"
        with gzip.open(path,'rt') as f:rows=json.load(f)['rows']
        last=[r['planning'][-1] for r in rows]
        e=dict(label=s['label'],opponent=s['opponent'],games=len(rows),
            shared_plot_days=st.fmean(r['shared_service_plots'] for r in last),
            shared_games=sum(r['shared_service_plots']>0 for r in last),
            regret_trials=st.fmean(r['regret_trials'] for r in last),
            regret_improvements=st.fmean(r['regret_improvements'] for r in last),
            cash=s['own_cash']['cash'],opponent_cash=s['rival_cash']['cash'],wages=s['own_cash']['hired'],
            labor_days=st.fmean(sum(d['max_hands'] for d in r['planning']) for r in rows),
            acquired=s['own_production']['acquired'],planted=s['own_production']['planted'],placed=s['own_production']['placed'],
            attempts=s['own_production']['attempts'],no_effect=s['own_production']['no_effect'],
            escaped=s['own_production']['escaped'],environment_loss=s['own_production']['environment_loss'],
            eod_loss=s['own_production']['eod_loss'],unassigned_jobs_not_comparable=s['planner_mean_unassigned_tasks'],
            cash_ledger=s['own_cash'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        result.append(e)
    (out/'summary.json').write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_DIAGNOSTIC',build=receipt['build'],results=result,
        caveat='Shared plot-days count planned multi-worker assignments, not guaranteed useful handoffs. Regret counters include hiring feasibility probes. Decomposed job counts cannot be compared as equal units.'),indent=2))
    lines=['# 共享维护与联合排班诊断','','| 配置 | 对手 | 现金 | 工资 | 雇工人日 | 多人同格计划日 | 替代排班采用次数 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in result:lines.append(f"| {r['label']} | {r['opponent']} | {r['cash']:,.0f} | {r['wages']:,.0f} | {r['labor_days']:.1f} | {r['shared_plot_days']:.1f} | {r['regret_improvements']:.1f} |")
    lines+=['','排班采用次数含预估雇工时的可行性探针，不是实际重排次数。拆分后一个Job含义改变，不把减少/增加Job数当作经济成功。该审计不增加独立强度样本。']
    (out/'TABLES_ZH.md').write_text('\n'.join(lines),encoding='utf8');print('\n'.join(lines))
if __name__=='__main__':main()
