"""Causal past-only public sale-time forecast; future transactions are labels only."""
from pathlib import Path
from collections import defaultdict
import gzip,json,statistics,sys
HERE=Path(__file__).resolve().parent
ITEMS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
def load(p):return json.loads(gzip.decompress(p.read_bytes()))
def distribution(hist):
    p=[0.]*24;p[1]=p[17]=.5
    # One fallback day plus up to three last observed sale-day distributions.
    for v in hist[-3:]:
        n=sum(v)
        if n:
            for h in range(24):p[h]+=v[h]/n
    return [x/(1+len(hist[-3:])) for x in p]
def earth(a,b):
    x=y=v=0
    for h in range(24):x+=a[h];y+=b[h];v+=abs(x-y)
    return v
def main():
    out=HERE/'sale_clock_calibration'
    if '--report-only' in sys.argv:
        summary=json.loads((out/'RESULTS.json').read_text())['summary']
        return report(out,summary)
    buckets=defaultdict(list);case_rows=[]
    for path in sorted((HERE/'public_ledger_audit_v2').glob('*.json.gz')):
        d=load(path);key=path.name.removesuffix('.json.gz');audit=load(HERE/'baseline_audit'/key/'AUDIT.json.gz')
        row=audit['source_row'];opp=row['opponent_seat'];group='win_control' if row['r2_win'] else 'loss'
        actual=defaultdict(lambda:[0]*24);observed=defaultdict(lambda:[0]*24);own=defaultdict(lambda:[0]*24)
        for tx in audit['transactions']:
            if tx['op']=='SELL' and tx['item'] in ITEMS[1:8]:
                target=actual if tx['seat']==opp else own
                target[(tx['step']//24,ITEMS.index(tx['item']))][tx['step']%24]+=tx['quantity']
        for f in d['frames']:
            s=f['source_step']
            if s<0:continue
            for i in range(1,8):
                if f['valid'][i] and f['rival_net'][i]>0:observed[(s//24,i)][s%24]+=f['rival_net'][i]
        histories=defaultdict(list);rows=[]
        for day in range(30):
            for i in range(1,8):
                # All observations used here strictly precede this day.
                if day and sum(observed[(day-1,i)]):histories[i].append(observed[(day-1,i)])
                y=actual[(day,i)];n=sum(y)
                if not n:continue
                a=[v/n for v in y];old=distribution([]);new=distribution(histories[i]);ours=own[(day,i)];nr=sum(ours)
                prior=lambda p:sum(p[h]*(sum(ours[h+1:])+.5*ours[h]) for h in range(24))/nr if nr else None
                r=dict(day=day,item=ITEMS[i],history_sale_days=min(3,len(histories[i])),actual_units=n,
                       old_transport_error=earth(old,a),new_transport_error=earth(new,a),
                       old_mean_hour=sum(h*p for h,p in enumerate(old)),new_mean_hour=sum(h*p for h,p in enumerate(new)),actual_mean_hour=sum(h*p for h,p in enumerate(a)))
                if nr:
                    r.update(actual_rival_early_share=prior(a),old_share_error=abs(prior(old)-prior(a)),new_share_error=abs(prior(new)-prior(a)))
                buckets[(group,ITEMS[i])].append(r);rows.append(r)
        case_rows.append(dict(case=key,rows=rows))
    summary={}
    for key,rows in buckets.items():
        joint=[r for r in rows if 'old_share_error' in r]
        summary['/'.join(key)]=dict(sale_days=len(rows),days_with_history=sum(r['history_sale_days']>0 for r in rows),
            old_time_distribution_error=statistics.mean(r['old_transport_error'] for r in rows),new_time_distribution_error=statistics.mean(r['new_transport_error'] for r in rows),
            joint_sale_days=len(joint),old_early_share_mae=statistics.mean(r['old_share_error'] for r in joint) if joint else None,
            new_early_share_mae=statistics.mean(r['new_share_error'] for r in joint) if joint else None)
    out=HERE/'sale_clock_calibration';out.mkdir(exist_ok=True)
    (out/'RESULTS.json').write_text(json.dumps(dict(cases=len(case_rows),summary=summary,rows=case_rows,
        boundary='Only previously certified public sales are predictors. Current-day actual opponent sales and own sales are evaluation labels, not online features. Sale/no-sale occurrence is NOT predicted here.'),indent=2))
    report(out,summary)
def report(out,summary):
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    lines=['# 历史公开成交时钟校准','',
           '只使用当天开始之前可从公开信息准确推断的成交。最近3个有成交记录的日内分布加1份旧先验（hour1/17各半）。无历史时完全回退原先验。',
           '未来实际成交及我方当天销售只用于评价；没有供给数量预测，也没有验证使用这个时钟就能提升胜率。','',
           '|商品|有销售的日数|有可用历史|原时刻分布误差|新误差|原先卖比例MAE|新MAE|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for i in ITEMS[1:8]:
        x=summary['loss/'+i]
        lines.append(f"|{i}|{x['sale_days']}|{x['days_with_history']}|{x['old_time_distribution_error']:.2f}|{x['new_time_distribution_error']:.2f}|{fmt(x['old_early_share_mae'])}|{fmt(x['new_early_share_mae'])}|")
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
