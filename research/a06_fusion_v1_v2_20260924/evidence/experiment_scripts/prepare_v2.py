"""Small rules-only ablation generation after a documented V1 holdout failure."""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'ACCEPTANCE.json').read_text())['status']=='NOT_MET'
    definitions=[
        ('v2_q5',{},5,'Smaller opening market order; same transparent liquidity guards.'),
        ('v2_q8',{},8,'Smaller opening market order; same transparent liquidity guards.'),
        ('v2_q15',{},15,'Larger opening order, still subject to available cash and warehouse capacity.'),
        ('v2_animal08',{'animal_bias':.8},10,'Discount marginal animal acquisition ranking; keep all crop choices and live planning.'),
        ('v2_pressure',{'batch_delivery':2},10,'Enable existing public-price-pressure mid-route delivery trigger.'),
        ('v2_h10',{'max_hands':10},10,'Lower labor cap to test whether marginal hiring costs exceed realized returns.'),
    ]
    manifest=json.loads((HERE/'CANDIDATES.json').read_text())
    known={r['id'] for r in manifest}
    for ident,config,q,reason in definitions:
        if ident in known:continue
        source=HERE/'candidates/cf_liq_h12_nointraday';dest=HERE/'candidates'/ident
        shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','build'))
        cfg=json.loads((dest/'policy/config.json').read_text());cfg.update(config)
        (dest/'policy/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        (dest/'overlay.json').write_text(json.dumps({'opening_liquidity':q},indent=2)+'\n')
        if q>10:
            text=(dest/'main.py').read_text();assert 'requested = min(10,' in text
            (dest/'main.py').write_text(text.replace('requested = min(10,','requested = min(15,'))
        manifest.append(dict(id=ident,parent='cf_liq_h12_nointraday',config_changes=config,opening_quantity=q,reason=reason))
    (HERE/'CANDIDATES.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'new_ids':[r[0] for r in definitions],'controls':['cf_liq_h12_nointraday','cf_liq_nointraday']}))


if __name__=='__main__':main()
