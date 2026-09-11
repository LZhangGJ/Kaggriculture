"""Freeze follow-up arms before observing the new development seeds."""
from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
configs=json.loads((EXP/'s3b_transfer_configs.json').read_text())
keys=['S3C03_control_hold0','S3C03_control_hold0.75',
      'S3C03_portfolio_layout_hold0','S3C03_portfolio_layout_hold0.75',
      'S1_control_hold0','S1_control_investment075_hold0','S1_control_investment075_hold0.75',
      'S1_portfolio_layout_investment075_hold0','S1_portfolio_layout_investment075_hold0.75']
for name,data in (
    ('s3b_followup_configs.json',{k:configs[k] for k in keys}),
    ('s3b_cash_configs.json',{'S3C03':configs[keys[0]],'S3C03_hold075':configs[keys[1]],
                            'S3C03_portfolio_layout':configs[keys[2]],'S1':configs['S1_control_hold0']})):
    p=EXP/name;assert not p.exists();p.write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(name,len(data))
