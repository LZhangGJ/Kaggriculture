"""Follow the observed animal-ranking improvement with one dose and two combinations."""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent


def main():
    definitions=[('v2_animal06',{'animal_bias':.6}),
                 ('v2_animal08_pressure',{'animal_bias':.8,'batch_delivery':2}),
                 ('v2_animal06_pressure',{'animal_bias':.6,'batch_delivery':2}),
                 ('v2_animal04',{'animal_bias':.4}),
                 ('v2_animal05',{'animal_bias':.5}),
                 ('v2_animal055',{'animal_bias':.55}),
                 ('v2_animal065',{'animal_bias':.65}),
                 ('v2_animal07',{'animal_bias':.7})]
    manifest=json.loads((HERE/'CANDIDATES.json').read_text());known={r['id'] for r in manifest}
    for ident,config in definitions:
        if ident in known:continue
        source=HERE/'candidates/cf_liq_h12_nointraday';dest=HERE/'candidates'/ident
        shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc','build'))
        cfg=json.loads((dest/'policy/config.json').read_text());cfg.update(config)
        (dest/'policy/config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        manifest.append(dict(id=ident,parent='cf_liq_h12_nointraday',config_changes=config,opening_quantity=10,
                             reason='Followup to improved animal08: one stronger discount and interaction with public-price-pressure delivery.'))
    (HERE/'CANDIDATES.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([x[0] for x in definitions]))


if __name__=='__main__':main()
