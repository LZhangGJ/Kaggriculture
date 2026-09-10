from pathlib import Path
from collections import defaultdict,Counter
import gzip,json,statistics,math,argparse
import run_panel as panel
HERE=Path(__file__).resolve().parent;OUT=HERE/'cash_projection_audit'

def metrics(rows):
    def stat(values):
        s=sorted(values)
        return dict(mean=statistics.mean(s),median=statistics.median(s),mae=statistics.mean(abs(x) for x in s),p10=s[int(.1*(len(s)-1))],p90=s[int(.9*(len(s)-1))]) if s else None
    return dict(days=len(rows),own_cash_error=stat([r['own_error'] for r in rows]),rival_cash_error=stat([r['rival_error'] for r in rows]),own_gain=stat([r['own_gain'] for r in rows]),rival_gain=stat([r['rival_gain'] for r in rows]),
                same_units=sum(r['units_same'] for r in rows),same_market_orders=sum(r['market_same'] for r in rows),same_whole_day=sum(r['all_same'] for r in rows),
                holdings_l1=statistics.mean(r['holdings_l1'] for r in rows),price_l1=statistics.mean(r['price_l1'] for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
    receipt=json.loads((OUT/('PILOT.json' if args.pilot else 'RESULTS.json')).read_text());rows=[]
    for r in receipt['rows']:
        data=json.loads(gzip.decompress((OUT/(r['key']+'.json.gz')).read_bytes()))
        assert data['status']=='PASS'
        for d in data['daily']:
            q=d['candidates'][d['winner']];a=d['actual'];cmp=d['action_comparison']
            rows.append(dict(key=r['key'],day=d['day'],lost=not data['row']['r2_win'],own_error=q['own_cash']-a['own_cash'],rival_error=q['rival_cash']-a['rival_cash'],own_gain=a['own_cash']-d['initial_own_cash'],rival_gain=a['rival_cash']-d['initial_rival_cash'],
                             units_same=cmp['units']['differences']==0,market_same=cmp['market']['differences']==0,all_same=cmp['all']['differences']==0,first_difference=cmp['all']['first'],
                             holdings_l1=sum(abs(x-y) for x,y in zip(q['holdings'],a['holdings'])),price_l1=sum(abs(x-y) for x,y in zip(q['prices'],a['prices'])),
                             predicted=q,actual=a,initial_own_cash=d['initial_own_cash'],initial_rival_cash=d['initial_rival_cash']))
    groups=defaultdict(list)
    for r in rows:
        group='loss' if r['lost'] else 'win_control';groups[group].append(r)
        stage='early' if r['day']<7 else 'middle' if r['day']<20 else 'late';groups[group+'_'+stage].append(r)
        groups['same_units' if r['units_same'] else 'different_units'].append(r)
    result=dict(status='DESCRIPTIVE_PREDICTION_AUDIT_NOT_CAUSAL_CANDIDATE_VALUE',cases=len(receipt['rows']),days=len(rows),overall=metrics(rows),groups={k:metrics(v) for k,v in groups.items()},
                largest_own_errors=sorted(rows,key=lambda r:abs(r['own_error']),reverse=True)[:10],largest_rival_errors=sorted(rows,key=lambda r:abs(r['rival_error']),reverse=True)[:10])
    name='PILOT_SUMMARY' if args.pilot else 'SUMMARY';panel.save(OUT/(name+'.json'),result)
    lines=['# 原R2：24步条件现金预测与真实回款核查','',f"原策略逐步一致通过；{result['cases']}局、{len(rows)}个日决策。误差=预测减实际，正数为高估。不是候选因果增益验证。",'',
           '| 分组 | 天观测 | 我方现金MAE | 我方偏差 | 对手现金MAE | 对手偏差 | 整日单位动作一致 |', '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in result['groups'].items():
        lines.append(f"| {k} | {v['days']} | {v['own_cash_error']['mae']:.1f} | {v['own_cash_error']['mean']:+.1f} | {v['rival_cash_error']['mae']:.1f} | {v['rival_cash_error']['mean']:+.1f} | {v['same_units']}/{v['days']} |")
    lines += ['', '口径：原规划器已进行的候选短模拟，只增加遥测；实际标签由冻结的官方完整原轨迹生成。模型不接收未来。',
              '对手短模拟只有基于当前公开产业的假设销售/饲料买入，没有其真实扩张投资，所以现金误差不能直接等同销售预测误差。',
              '同日单位动作相同而现金不同，仍可能来自预测价格/共享市场成交；日末商品留存不等同产量失效。',
              '绝对预测误差不等同候选排序误差：候选间共通偏差可能相互抵消。本审计只定位问题，不凭此宣称修改必然上分。','']
    (OUT/(name+'_ZH.md')).write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('largest')}),flush=True)
if __name__=='__main__':main()
