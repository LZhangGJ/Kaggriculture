from pathlib import Path
import copy,json
EXP=Path(__file__).resolve().parents[1]
configs=json.loads((EXP/'s3b_feed_configs.json').read_text())
keys=['S3C03','cover1_cap60_demand0','cover1_cap60_demand0.5','cover7_cap30_demand0.5','cover7_cap60_demand1']
out={k:configs[k] for k in keys}
p=EXP/'s3b_feed_followup.json';assert not p.exists();p.write_text(json.dumps(out,indent=2),encoding='utf-8')
new={'S3C03':configs['S3C03']}
for horizon in (1,3,7):
    for cover in (3,7):
        for demand in (0.,.5,1.):
            cfg=copy.deepcopy(configs['S3C03'])
            cfg.update(feed_forecast_days=horizon,feed_cover_days=cover,own_feed_demand_weight=demand)
            new[f'forecast{horizon}_cover{cover}_demand{demand:g}']=cfg
p=EXP/'s3b_feed_forecast_configs.json';assert not p.exists();p.write_text(json.dumps(new,indent=2),encoding='utf-8')
print(len(out),len(new))
