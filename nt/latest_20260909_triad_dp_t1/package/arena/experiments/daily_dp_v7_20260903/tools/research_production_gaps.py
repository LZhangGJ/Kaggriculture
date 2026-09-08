"""Disentangle recorded output versus conversion losses in existing games."""
from pathlib import Path
import gzip,hashlib,json,statistics as st
E=Path(__file__).resolve().parents[1];out=E/'receipts/s4x_production_gap_v2';out.mkdir(exist_ok=False)
names=['小麦','胡萝卜','番茄','草莓','瓜','蛋','奶','羊毛','肥料'];results=[];hashes={}
for label,folder,variant in [('shared','s4m1_pool_audit_N50_v1','all_intraday_insert'),('autonomous','s4u_pool_audit_N50_v1','full_chain_autonomous')]:
 for opp in ('g001','g003','boatlee_v29','kaito_v58','lynn_v5','yhay81_six_day','yhay81_three_day','ecobot_v7'):
  path=E/'receipts'/folder/f'{variant}_{opp}.json.gz';hashes[str(path.relative_to(E))]=hashlib.sha256(path.read_bytes()).hexdigest();rows=json.load(gzip.open(path,'rt'))['rows'];assert len(rows)==100
  for item in (3,4,6,7):
   sides=[]
   for side in (0,1):
    totals=[]
    for row in rows:
     seat=row['seat'] if side==0 else 1-row['seat'];days=[d[seat] for d in row['production']]
     d={k:sum(x[k][item] for x in days) for k in ('generated','acquired','used','unit_discard','environment_loss','drop_loss','eod_loss','bought','sold')}
     d['terminal']=days[-1]['end_field'][item]+days[-1]['end_private'][item]
     d['loss']=sum(d[k] for k in ('unit_discard','environment_loss','drop_loss','eod_loss'))
     assert d['generated']+d['bought']==d['sold']+d['used']+d['loss']+d['terminal'],(opp,row['seed'],seat,item,d)
     start_key,start_index=('planted',item) if item<5 else ('placed',1 if item==6 else 2)
     starts=[x[start_key][start_index] for x in days]
     d['starts']=sum(starts);d['weighted_start_total']=sum(day*q for day,q in enumerate(starts))
     d['early_starts']=sum(starts[:10])
     totals.append(d)
    sides.append({k:st.fmean(x[k] for x in totals) for k in totals[0]})
   a,b=sides;results.append(dict(background=label,opponent=opp,item=names[item],own=a,rival=b,sold_gap=a['sold']-b['sold'],generated_gap=a['generated']-b['generated'],loss_gap=a['loss']-b['loss'],terminal_gap=a['terminal']-b['terminal']))
(out/'summary.json').write_text(json.dumps(dict(results=results,hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),games=1600,conservation_pass=True),ensure_ascii=False,indent=2))
lines=['# 售出量缺口：产出与转化分开核账','','复用1600完整局，均通过 商品产出+买入=卖出+使用+损失+终局残余。这里只记录实际增加到田间的产品；达到储量上限后未能生成的潜在产量不在generated中，因此不能由此断言问题仅在投资、与收获调度无关。','','## 共享版草莓（每对手100局均值）','','|对手|实际生成 我/他|卖出 我/他|损失 我/他|终局残余 我/他|','|---|---:|---:|---:|---:|']
for r in results:
 if r['background']=='shared' and r['item']=='草莓':
  a,b=r['own'],r['rival'];lines.append(f"|{r['opponent']}|{a['generated']:.2f}/{b['generated']:.2f}|{a['sold']:.2f}/{b['sold']:.2f}|{a['loss']:.2f}/{b['loss']:.2f}|{a['terminal']:.2f}/{b['terminal']:.2f}|")
lines+=['','## 草莓投产数量与时点（day index 从0计）','','数量包括补种；加权日期混合了不同批次，不应照抄成固定种植日。产出/种植量还混合布局、维护、剩余生产周期等因素。','','|对手|种植次数 我/他|前10天种植 我/他|加权种植日 我/他|产出/种植 我/他|','|---|---:|---:|---:|---:|']
for r in results:
 if r['background']=='shared' and r['item']=='草莓':
  a,b=r['own'],r['rival'];lines.append(f"|{r['opponent']}|{a['starts']:.2f}/{b['starts']:.2f}|{a['early_starts']:.2f}/{b['early_starts']:.2f}|{a['weighted_start_total']/a['starts']:.2f}/{b['weighted_start_total']/b['starts']:.2f}|{a['generated']/a['starts']:.2f}/{b['generated']/b['starts']:.2f}|")
lines+=['','这里的损失不自动归为硬错误，可能有计划性舍弃。流水数量分解不是因果利润估计，不能把对手多产的量当成我方可免费获得的量。']
(out/'TABLES_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines))
