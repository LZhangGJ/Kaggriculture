"""Unify baseline evidence, verifying old protocol/R2/engine hashes first."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_panel as panel

new_pool = json.loads((HERE/'pool_new7.json').read_text())
old_pool = [
    dict(id='shop0909',working='research/shop_router_0909_review_20260909'),
    dict(id='aurax_shop_v2',working='research/reactive_rescue_review_20260909/aurax7/working'),
    dict(id='seven_turn',working='research/reactive_rescue_review_20260909/seven_turn/working'),
    dict(id='ahmed_v27',working='research/ahmed_reactive_review_20260909/working'),
]
panel.save(HERE/'pool_all11.json', new_pool + old_pool)
rows = []
for name in ['baseline_soil_moon','baseline_remaining5']:
    assert json.loads((HERE/name/'RESULTS.json').read_text())['overall']['errors'] == 0
    for row in json.loads((HERE/name/'rows.json').read_text()):
        row['trace'] = str((HERE/name/row['trace']).relative_to(ROOT))
        rows.append(row)

origin = ROOT/'experiments'
sources = [
    ('shop0909', origin/'shop_router_0909_vs_r2_200_20260909', ''),
    ('aurax_shop_v2', origin/'reactive_rescue_vs_r2_200_20260909', 'aurax7'),
    ('seven_turn', origin/'reactive_rescue_vs_r2_200_20260909', 'seven_turn'),
    ('ahmed_v27', origin/'ahmed_v27_vs_r2_200_20260909', ''),
]
proofs = []
for name, folder, sub in sources:
    protocol = json.loads((folder/'PROTOCOL.json').read_text())
    assert protocol['seeds'] == list(range(2609110000,2609110100))
    assert protocol.get('official_version') == '1.32.7'
    expected_r2 = protocol.get('r2_binary_sha256')
    if expected_r2 is None:
        expected_r2 = next(v for k,v in protocol['source_hashes'].items() if k.endswith('policy/agent.so'))
    assert expected_r2 == panel.sha(panel.old.R2/'agent.so')
    assert protocol.get('r2_config_sha256',protocol.get('r2_config_hash')) == panel.sha(panel.old.R2/'config.json')
    assert protocol.get('r2_wrapper_sha256',protocol.get('r2_wrapper_hash')) == panel.sha(panel.old.R2/'agent.py')
    assert protocol.get('official_source_sha256',protocol.get('official_source_hash')) == panel.sha(panel.old.REF/'official/kaggriculture.py')
    working = ROOT/next(x['working'] for x in old_pool if x['id']==name)
    if name=='shop0909':
        for path,digest in protocol['source_hashes'].items():
            if '/policy/' not in path:
                assert panel.sha(working/Path(path).name)==digest
    elif name in {'aurax_shop_v2','seven_turn'}:
        receipt_name = 'aurax7' if name=='aurax_shop_v2' else name
        for fn,h in protocol['original_source_receipts'][receipt_name]['file_sha256'].items():
            assert panel.sha(working/fn)==h
    else:
        assert panel.sha(working/'main.py') == protocol['original_source_receipt']['main_sha256']
    old_rows = json.loads((folder/sub/'rows.json').read_text())
    assert len(old_rows)==200
    for r in old_rows:
        assert not r['runtime_error'] and r['steps']==719
        rows.append(dict(opponent=name,seed=r['seed'],opponent_seat=r['shop_seat'],steps=719,
                         r2_cash=r['r2_cash'],opponent_cash=r['shop_cash'],r2_margin=-r['shop_margin'],
                         r2_win=r['r2_win'],opponent_win=r['shop_win'],tie=r['tie'],runtime_error=None,
                         shops=r['shops'],joint_action_sha256=r['joint_action_sha256'],
                         trace=str((folder/sub/r['trace']).relative_to(ROOT))))
    proofs.append(dict(opponent=name,status='MATCHED_REUSED',protocol=str((folder/'PROTOCOL.json').relative_to(ROOT))))
assert len(rows)==2200
by = {x['id']:panel.summarize([r for r in rows if r['opponent']==x['id']]) for x in new_pool+old_pool}
panel.save(HERE/'BASELINE_ALL11.json',dict(games=2200,new_games=1400,reused_games=800,
                                        by_opponent=by,overall=panel.summarize(rows),reuse_proofs=proofs))
panel.save(HERE/'baseline_rows_all11.json',rows)
report = ['# R2 11 公开对手基线验收（2026-09-10）','',
          '每对手100相同种子×双座位200局。双方实时行动，官方1.32.7；R2原C++发布二进制；对手原Python。',
          '新7个1,400局，旧4个800局经版本、R2配置及官方规则哈希核对后复用。不把旧局冒充新局。','',
          '| 对手 | R2胜/负/平 | R2胜率 | R2平均现金 | 对手平均现金 | 平均分差 |',
          '|---|---|---:|---:|---:|---:|']
for name,v in by.items():
    report.append(f"| {name} | {v['r2_wins']}/{v['losses']}/{v['ties']} | {v['r2_win_rate']:.1%} | {v['r2_mean_cash']:,.0f} | {v['opponent_mean_cash']:,.0f} | {v['mean_margin']:+,.0f} |")
a=panel.summarize(rows)
report += ['',f"总计{a['r2_wins']}/2200胜，等权平均胜率{a['r2_win_rate']:.2%}。异常0，全部719步；双座位高度相关，不视为200个独立随机剧本。",'',
           '这是冻结原R2的开发基线，不是修复后最终验收。独立最终种子尚未使用。',
           '新7个已分别通过首种子双座位与同进程重新导入复跑动作哈希一致性；源包和Notebook见research/public_opponents_r2_20260910。',
           '本地计时不代表Kaggle官方容器配额/超时检查。', '',
           '下一步按开局、商店、批次回款、扩张/用工、转产、物流、终局分组复盘；先测试采购融资修复，统计败转胜与胜转败。']
(HERE/'BASELINE_ACCEPTANCE_ZH.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print(json.dumps(dict(overall=a,by_opponent=by),ensure_ascii=False),flush=True)
