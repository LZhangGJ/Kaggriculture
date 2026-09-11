"""Retain all evidence; do not promote an unsuccessful new parameter set."""
from pathlib import Path
import hashlib,json,random,shutil,statistics
EXP=Path(__file__).resolve().parents[1];ROOT=EXP.parents[1]
load=lambda name:json.loads((EXP/'receipts'/name/'results.json').read_text())
old=load('s3_review_new50');reg=load('s3b_legacy_regression')
signature=lambda rows:{(r['label'],r['opponent'],r['seed'],r['seat']):(r['cash'],r['opponent_cash'],r['win']) for r in rows}
assert signature(old['rows'])==signature(reg['rows']), 'legacy regression'
dev=load('s3b_transfer_dev50');fresh=load('s3b_followup_new50')
feed=load('s3b_feed_dev50');feed_new=load('s3b_feed_new50')
forecast=load('s3b_feed_forecast_dev50')
get=lambda data,label:next(x for x in data['summaries'] if x['label']==label)
cash=json.loads((EXP/'receipts/s3b_cash_audit_dev50/summary.json').read_text())
feed_cash=json.loads((EXP/'receipts/s3b_feed_cash_dev50/summary.json').read_text())
base_cash=next(x for x in cash['summaries'] if x['label']=='S3C03' and x['opponent']=='g001')
build=json.loads((EXP/'native/build/build_receipt.json').read_text())
out=EXP/'profiles/s3b';out.mkdir(exist_ok=True)
assert not (out/'stage_receipt.json').exists(), 'already archived'
for rel,h in build['source_hashes'].items():
    p=EXP/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
for file in EXP.glob('s3b*json'):shutil.copy2(file,out/file.name)
reference=ROOT/'gpt_review/gpt_code/my_agent_v7.py';ref_hash=hashlib.sha256(reference.read_bytes()).hexdigest()
assert ref_hash=='8147655aa8a3d0bee3a130352272488143c6d0e89decd922cebbe9a4a2825db6'
shutil.copy2(reference,out/'reviewed_gpt_v7.py')
combined=[]
for data in (dev,fresh):combined += [r for r in data['rows'] if r['label']=='S3C03_control_hold0']
g=[r for r in combined if r['opponent']=='g001'];pas=[r for r in combined if r['opponent']=='pass']
clusters=[statistics.fmean(r['win'] for r in g if r['seed']==s) for s in sorted({r['seed'] for r in g})]
rng=random.Random(903711);draws=sorted(statistics.fmean(rng.choices(clusters,k=len(clusters))) for _ in range(10000))
stats=dict(games=len(g),independent_seeds=len(clusters),wins=sum(r['win'] for r in g),pass_mean=statistics.fmean(r['cash'] for r in pas),own_mean=statistics.fmean(r['cash'] for r in g),margin=statistics.fmean(r['margin'] for r in g),seed_cluster_bootstrap95=[draws[250],draws[9749]])
official=[]
for name in ('s3b_hold_official','s3b_feed_forecast_official'):
    d=json.loads((EXP/'receipts'/name/'acceptance.json').read_text());assert d['status']=='PASS';official.append(dict(receipt=name,games=len(d['rows'])))
mechanisms=json.loads((EXP/'receipts/s3b_mechanisms_final/acceptance.json').read_text());assert mechanisms['status']=='PASS'
receipt=dict(stage='S3b development, NOT acceptance',preferred_candidate='S3C03',new_default_parameters_promoted=False,combined_development=stats,regression_games=len(reg['rows']),cash_audit_games=sum(s['games'] for s in cash['summaries'])+sum(s['games'] for s in feed_cash['summaries']),all_cash_steps_reconciled=True,official=official,mechanism_checks=mechanisms['native_mechanism_checks'],build=build,reference_sha256=ref_hash,formal_holdouts_used=False)
(out/'stage_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
lines=['# S3b：借鉴GPT v7、现金归因与饲料消融','',
       '日期：2026-09-03。结论：能力已接入C++并验证，**本轮没有足够证据晋级新配置；优先版本仍为S3C03，双目标未完成。**','',
       '## 1. 借鉴与边界','',
       '- 新增持货估值的公开对手供给折扣；投资评分、短期主动出售、融资/腾仓的出售排序分别控制。默认权重0保持原行为。',
       '- 将3羊1牛、7瓜5麦3草莓作为独立开局候选；开局地块分配顺序也独立参数化。没有写入逐步高手日历或对手身份判断。',
       '- 沿用已修复的C++资源降级/腐坏处理，不复制外部Python版仍有的执行缺口。供给估计考虑公开动物饲料需求，因而不是原GPT文件的逐字移植。',
       '- “4/10”没有原始测试记录。本轮是思想迁移消融，**不是对原my_agent_v7.py的4/10复现或否定**。',
       '- 补充可选短期备料价格预测：用已知商店、未来商店期望、公开产能及我方现有粮食预测；不是查询真实未来。默认关闭，未晋级。','',
       '## 2. 同批对照与新seed复查','',
       'A=20261401–20261450；B=20261501–20261550。每批50个seed双座位100局；PASS与完整G001分别跑。B的短名单先冻结后测试，后续已用于分析，现同属开发集。','',
       '| 配置 | A胜局/100 | B胜局/100 | A静止现金 | B静止现金 |','|---|---:|---:|---:|---:|']
for label,title in [('S3C03_control_hold0','原S3C03'),('S3C03_control_hold0.75','仅新增持货供给0.75'),('S3C03_portfolio_layout_hold0','仅换GPT开局/布局'),('S3C03_portfolio_layout_hold0.75','GPT开局/布局+持货供给')]:
    a=get(dev,label);b=get(fresh,label);lines.append(f"| {title} | {a['g001']['wins']} | {b['g001']['wins']} | {a['pass']['mean_cash']:,.2f} | {b['pass']['mean_cash']:,.2f} |")
lines += ['',f"原配置合并开发集：{stats['wins']}/{stats['games']}胜，静止均值{stats['pass_mean']:,.2f}；按seed聚类bootstrap95%区间约{stats['seed_cluster_bootstrap95'][0]:.1%}–{stats['seed_cluster_bootstrap95'][1]:.1%}。不是正式验收，不把一批的18万视作总体或双目标达标。",'',
          '## 3. 已闭合的现金账','',
          'C++离线根据实际成交量逐条重建收入/支出，保留双方同索引逐单位同时报价、1元售出不增加市场库存等规则；每一步、每天和终局现金均对账。审计数据没有进入策略View，不能用于线上偷看。','',
          '| 项目：A批对G001每局平均 | S3C03 | 同局G001 |','|---|---:|---:|']
x=base_cash['own'];y=base_cash['rival']
for title,u,v in [('小麦购买支出',x['products'][0],y['products'][0]),('小麦购买量',x['bought_products'][0],y['bought_products'][0]),('草莓销售收入',x['sales'][3],y['sales'][3]),('草莓销售量',x['sold'][3],y['sold'][3]),('雇工支出',x['hired'],y['hired']),('终局现金',x['cash'],y['cash'])]:lines.append(f'| {title} | {u:,.2f} | {v:,.2f} |')
lines += ['',f"小麦支出差{x['products'][0]-y['products'][0]:,.2f}，草莓销售收入差{x['sales'][3]-y['sales'][3]:,.2f}。这是会计差额，不是保证能补回的利润；必须区分数量、售价、自用、库存、资本和工时。",'',
          '## 4. 饲料实验：不能只看省钱','',
          '测试既有通用参数：备1/3/7天、上限30/60/90、我方饲料需求估值权重0/0.5/1。保持原开局与其余修复。','',
          '| 配置 | A胜局/100 | B胜局/100 | A静止现金 | B静止现金 |','|---|---:|---:|---:|---:|']
for label in ('S3C03','cover1_cap60_demand0','cover1_cap60_demand0.5','cover7_cap30_demand0.5','cover7_cap60_demand1'):
    a=get(feed,label);b=get(feed_new,label);lines.append(f"| {label} | {a['g001']['wins']} | {b['g001']['wins']} | {a['pass']['mean_cash']:,.2f} | {b['pass']['mean_cash']:,.2f} |")
short=next(s for s in feed_cash['summaries'] if s['label']=='cover1_cap60_demand0' and s['opponent']=='g001')['own']
lines += ['',f"A批只备一天时，小麦支出从{x['products'][0]:,.2f}降至{short['products'][0]:,.2f}，但我方现金从{x['cash']:,.2f}降至{short['cash']:,.2f}；牛奶/草莓销售收入同时下降，购买动物和草莓种子也减少。不是无损省钱。A批51胜并未在B批复现，不能据此晋级。",'',
          '另测1/3/7天短期价格预测，共19个配置，最高仍51/100且静止现金下降；一些短期预测实际退化为只备一天，不能当作独立新突破。该分支只完成A批及机制/官方语义验证，未证明新seed收益。','',
          '## 5. 验证与保存','',
          f"- 旧7个冻结配置，PASS与G001共{len(reg['rows'])}局，最终双方现金与胜负逐项与旧收据完全一致；新增关闭开关没有暗改原策略。",
          f"- 原生机制测试{mechanisms['native_mechanism_checks']}项通过；新增包含持货供给改变真实融资动作、已有粮食降低预测外购需求、对手养殖提高预测购粮压力。",
          '- 新持货开关和新备料预测各2局官方1.32.7逐步复验，共4×719步；双方状态和原Python G001动作一致，含真实切换局。不是新候选Python翻译验收。',
          f"- 现金审计{receipt['cash_audit_games']}局，双方每步现金闭合；另跑不带审计的相同对局确认结果不受统计影响。全部对局完整719步。",
          '- 新机制测试最初用了Day28却只有1家商店的不一致夹具，导致动作预期断言失败；修正为合法8家商店配置后通过。没有拿该失败夹具结果作收益证据。',
          '- 批量策略、模拟、现金统计均C++16线程；Python仅准备/汇总数据及4局官方复验。',
          '- 所有旧最优、负结果、新开关与参考源码均保存；profiles/s3b/stage_receipt.json包含源码/构建哈希。未提交、未推Git、未训练模型。',
          '- 正式PASS20261201–20261250和G00120261301–20261350仍未使用。','',
          '## 6. 后续方向','',
          '不再优先微调持货折扣或盲目减粮。下一步沿现金账追查自产小麦的产出→自用→外购→出售，以及草莓生产兑现的时间：目标是在不缩小有效生产、不破坏现金周转的前提下降低外购依赖。用单位/地块的实际执行记录区分没规划、没买成、没种下、没收获与主动取舍，再做单能力对照。不能根据G001身份或原路线写固定日期、数量来补分。','',
          '## 收据索引','']
for name in ('s3b_transfer_dev50','s3b_followup_new50','s3b_cash_audit_dev50','s3b_feed_dev50','s3b_feed_new50','s3b_feed_cash_dev50','s3b_feed_forecast_dev50','s3b_legacy_regression','s3b_hold_official','s3b_feed_forecast_official','s3b_mechanisms_final'):lines.append(f'- ../receipts/{name}/')
(EXP/'reports/S3B_TRANSFER_AND_CASH_AUDIT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False,indent=2))
