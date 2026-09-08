"""Minimal diagnostics; never changes submitted source or test policy settings."""
import copy
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
from cpu_runtime import LocalGame,load_agent,pass_agent

entry=load_agent(ROOT/'gpt_review/gpt_code/gpt-6-dp/gpt-6-kaggriculture_daily_dp_agent-v2.py')
game=LocalGame(61010)
probe=None
for step in range(719):
    obs=game.observation(0)
    action=entry(obs,copy.deepcopy(game.configuration))
    if obs['hour']==0:
        policy=entry.__globals__['_POLICY']
        for y,row in enumerate(obs['farms'][0]['tiles']):
            for x,tile in enumerate(row):
                if not isinstance(tile,dict) or tile.get('crop') not in ('STRAWBERRY','TOMATO'):
                    continue
                first=10 if tile['crop']=='STRAWBERRY' else 8
                if obs['day']-tile['planted_day']<first or tile.get('yield_units',0)==0:
                    continue
                # Compare a genuinely mature, nonempty observed crop to the same
                # public state with its held product removed. Forecast must see it.
                empty=copy.deepcopy(obs)
                empty['farms'][0]['tiles'][y][x]['yield_units']=0
                a=policy.forecast_base(obs)
                b=policy.forecast_base(empty)
                j=entry.__globals__['IX'][tile['crop']]
                probe=dict(seed=61010,seat=0,step=step,day=obs['day'],tile=[x,y],
                           actual_tile=tile,changed_yield=0,
                           total_crop_forecast_actual=float(a[0][:,j].sum()),
                           total_crop_forecast_changed=float(b[0][:,j].sum()),
                           arrays_equal=all(bool((v==w).all()) for v,w in zip(a,b)))
                break
            if probe:break
    if probe:break
    game.advance([action,pass_agent(game.observation(1),game.configuration)])

inventory=10000
quotes=[]
for _ in range(100):
    q=game.engine.market_price('WOOL',inventory)
    quotes.append(q)
    if q>1:inventory+=1
nominal=100*game.engine.market_price('WOOL',10000)
result=dict(mature_crop_prediction_probe=probe,
            wool_financing_example=dict(initial_inventory=10000,quantity=100,
                initial_quote=200,nominal_v2_cash_credit=nominal,
                official_pass_opponent_cash_credit=sum(quotes),
                overestimate=nominal-sum(quotes),
                note='Diagnostic of V2 cash projection formula, not an assertion this order occurred in its games'))
(HERE/'semantic_probes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
