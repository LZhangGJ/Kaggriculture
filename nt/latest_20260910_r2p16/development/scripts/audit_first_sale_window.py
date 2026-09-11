"""Paired-prefix intervention accounting, not a frozen-opponent experiment."""
from pathlib import Path
from collections import Counter
import argparse,json,gzip
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def gz(p):return json.loads(gzip.decompress(p.read_bytes()))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args();folder=args.folder.resolve()
    results=[]
    for r in json.loads((folder/'REVIEW_CASES.json').read_text()):
        b,n=r['baseline'],r['candidate'];own=1-b['opponent_seat'];start=r['first_own_difference'];end=((start+3)//4)*4+1
        key=f"{b['opponent']}_{b['seed']}_{b['opponent_seat']}"
        bt,nt=gz(ROOT/b['trace']),gz(folder/n['trace'])
        assert bt['actions'][:start]==nt['actions'][:start]
        before=gz(HERE/'baseline_audit'/key/'AUDIT.json.gz');after=gz(folder/'audit'/key/'AUDIT.json.gz')
        x=bt['actions'][start:end+1];y=nt['actions'][start:end+1]
        same_units=all([a[own]['farmer'],a[own]['hands']]==[c[own]['farmer'],c[own]['hands']] for a,c in zip(x,y))
        same_op=all(a[1-own]==c[1-own] for a,c in zip(x,y))
        def totals(a):
            amount=Counter();money=Counter()
            for t in a['transactions']:
                if start<=t['step']<=end:
                    k=f"{t['seat']}:{t['op']}:{t['item']}";amount[k]+=t['quantity'];money[k]+=t['cash_delta']
            return amount,money
        q0,c0=totals(before);q1,c1=totals(after)
        cash0=before['post_steps'][end]['cash'];cash1=after['post_steps'][end]['cash']
        differences={k:c1[k]-c0[k] for k in set(c0)|set(c1) if c0[k]!=c1[k]}
        results.append(dict(key=key,group=r['group'],start=start,end_inclusive=end,same_all_own_units=same_units,same_live_rival_actions=same_op,same_filled_transaction_quantities=q0==q1,
                            start_before=bt['actions'][start][own],start_after=nt['actions'][start][own],
                            own_cash_difference=cash1[own]-cash0[own],rival_cash_difference=cash1[1-own]-cash0[1-own],cash_account_differences=differences,
                            final_margin_change=n['r2_margin']-b['r2_margin']))
    panel.save(folder/'FIRST_SALE_WINDOW.json',results)
    lines=['# 首次动作偏离：短出售窗口的直接核对','',
           '只比较两条真实完整对战轨迹共同前缀后的首个窗口；双方原程序均实时响应。没有给策略真实未来。终局变化不能全归因于这几步价差。','',
           '| 对局 | 步窗口 | 我方单位相同 | 对手实际动作相同 | 成交量相同 | 窗口我方现金差 | 对手现金差 | 终局分差变化 |','|---|---|---|---|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['key']} | {r['start']}–{r['end_inclusive']} | {r['same_all_own_units']} | {r['same_live_rival_actions']} | {r['same_filled_transaction_quantities']} | {r['own_cash_difference']:+.0f} | {r['rival_cash_difference']:+.0f} | {r['final_margin_change']:+.0f} |")
    lines+=['','完整JSON列每项实际现金科目差额。只有动作/成交量控制相同时，才能把该窗口差额解释为同批货出售时点效应；否则只是轨迹分解。']
    (folder/'FIRST_SALE_WINDOW_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(results),flush=True)
if __name__=='__main__':main()
