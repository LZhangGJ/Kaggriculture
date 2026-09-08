"""Close the entire S4V factorial round, retaining failed strength/anomaly gates."""
from pathlib import Path
import hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];hashes={}
def read(rel):
    p=E/rel;hashes[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
run=read('receipts/s4v_execution_v1/acceptance.json');assert run['status']=='COMPLETE_DEVELOPMENT_ROUND_NOT_GOAL_ACCEPTANCE'
panel=read('receipts/s4v_twelveway_N50_v1/results.json');assert len(panel['rows'])==10800
audit=read('receipts/s4v_pool_audit_N50_v1/summary.json');assert audit['all_ledgers_reconciled'] and audit['unchanged_result_games']==8100
comparison=read('receipts/s4v_comparison_N50_v1/summary.json');assert comparison['unchanged_controls']==2700
mechanism=read('receipts/s4v_mechanism_v2/acceptance.json');assert mechanism['status']=='PASS' and mechanism['checks']==22
regression=read('receipts/s4v_escape_regression_v1/acceptance.json');assert len(regression['summary'])==4
build=run['build']
for rel,h in build['source_hashes'].items():assert hashlib.sha256((E/'profiles/s4v/source'/rel).read_bytes()).hexdigest()==h
official=[]
for prefix in ('g001','g003','boatlee','kaito','lynn','fieldbook','three_day','ecobot'):
    r=read(f'receipts/{prefix}_s4v_official_v1/acceptance.json');assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256'];official.append(prefix)
for prefix in ('boatlee','kaito','lynn','fieldbook','three_day','ecobot'):
    r=read(f'receipts/{prefix}_s4v_isolation_v1/acceptance.json');assert r['status']=='PASS' and r['build']['binary_sha256']==build['binary_sha256']
escaped=sum(sum(g['own_production']['escaped'])*g['games'] for g in audit['summary'])
ineffective=sum(sum(g['own_production']['no_effect'])*g['games'] for g in audit['summary'])
opps=('g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7')
success=[label for label,s in panel['summary'].items() if all(s[k]['win_rate']>.90 for k in opps)]
assert not success,'Unexpected strength change: requires independent gate review, not auto promotion'
out=E/'receipts/s4v_stage_acceptance_v1';out.mkdir(exist_ok=False)
receipt=dict(status='COMPLETE_STAGE_REJECTED_FOR_PROMOTION',input_hashes=hashes,build=build,
    full_panel_games=10800,ledger_repeat_games=8100,unchanged_controls=2700,no_effect_unit_actions=ineffective,
    unplanned_escaped_animals=escaped,official_opponents=official,strength_gate=False,anomaly_gate=not(escaped or ineffective),
    online_python_latency_gate='NOT_TESTED',promoted=False,holdout_used=False,final_goal_acceptance=False,
    next='S4W isolated revocable CARE/FEED alternatives, feasibility then conditional net cash; no identity/future/Oracle')
(out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
table=(E/'receipts/s4v_comparison_N50_v1/TABLES_ZH.md').read_text()
lines=['# S4V 完整复盘：建议落实、任务插入及长链组合','','## 结论','','不晋级。12配置×50开发seed×双座位×9对手（含PASS）=10,800局全部完成；8,100个新配置账本重复与面板双方终盘现金、溢仓完全一致。旧2,700对照逐局复现。没有任何同一配置对八个实时对手逐一超过90%。','','## 问题证据和通用改动','','1. 原市场观察审计发现日初队列/部分回仓路径没有落实已有出售建议；不是证明每次补卖都盈利。`continuous_market_execution`默认关，沿用原公开供给预测，不挤占十单、不改融资顺序、保护领料预留。','2. 原4个非计划漏喂局有人手排满、CARE仍排但FEED缺失。`insert_missing_feed`默认关，在全体工人的语义边界尝试插入，显式领料、预留和deadline约束，保留原任务。','3. 基线、共享调度、完整自主上下层三种背景，分别原样/市场/补喂/双开。全部组合实际执行，未用单项结果否定长链。','','## 实际强度','',''+table,'','## 配对证据，不用小幅涨分冒充稳定成功','']
for e in comparison['effects']:
    if (e['before']=='all_intraday_insert' and e['after'].endswith('_feed') and e['opponent']=='g003') or (e['before']=='full_chain_autonomous' and e['after'].endswith('_market') and e['opponent'] in ('g001','g003','kaito_v58')):
        w=e['estimates']['win'];m=e['estimates']['margin'];lines.append(f"- {e['before']} → {e['after']} / {e['opponent']}：胜率差{w['mean']*100:+.1f}个百分点，seed配对近似95%区间[{w['approx95'][0]*100:.2f}, {w['approx95'][1]*100:.2f}]；资金差改善{m['mean']:+.2f}。")
lines += ['','双开交互使用 `both - market - feed + base` 按seed计算，完整数据见comparison/summary.json；均属开发集探索，未校正多重比较，不能据局部区间宣布泛化。','',f'## 工程与失败门\n\n22新机制和既有回归通过；32官方完整局、6对手隔离通过。同一冻结build的8,100重复审计记录无效单位动作{ineffective:g}，动物逃跑{escaped:g}。原4个逃跑在所有组合中仍存在，不能说已经修好。原生48局时延探针最大约0.493秒；最终线上Python包未验收。','','## 本轮改变了什么，没改变什么','','市场执行衔接改变了成交及后续现金，但改善远小于整体差距，且Kaito出现倒退。仅补单不是完整商战能力；不能从本轮失败断言市场策略无用。补喂插入保留全部旧任务，在真实排满场景仍无可行位置，因而不能仅调其估值解决。','','## 下一步','','先做隔离S4W：允许明确撤回同一动物的CARE并重新分工，用当前观察构造的条件后果比较KEEP和替换，而非强制保动物。先验证真实失败状态里候选是否存在、估值是否已知、时延能否接受；不通过不启动大面板。该局部能力不是四项宏观能力的全部，不代表90%目标已解决。','','生产源码与所有分支保持可追溯；原基线不替换。开发N已反复用于选型，P未用，后续晋级必须新增未见seed。Goal保持active，未Git、未Kaggle。']
(E/'reports/S4V_FULL_REVIEW_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('build','input_hashes')}),flush=True)
