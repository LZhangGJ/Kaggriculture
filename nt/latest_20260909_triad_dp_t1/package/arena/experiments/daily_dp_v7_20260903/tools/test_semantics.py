"""Focused rule-boundary tests; does not substitute for free-running matches."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
from cpu_runtime import LocalGame


def main():
    spec=importlib.util.spec_from_file_location('v7_semantics',HERE/'agents/agent_v7.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    obs=LocalGame(7).observation(0)
    ctl=m.DailyDPController();ctl.day=10;ctl.target={(4,0):'STRAWBERRY'}
    tile={'kind':'PLANT','crop':'STRAWBERRY','planted_day':0,'yield_units':1,'max_lifespan_step':-1,
          'watered_today':False,'fertilized_until_day':50}
    obs['day']=10;obs['hour']=0;obs['step']=240;obs['farms'][0]['tiles'][0][4]=tile
    before=[a[0] for j in ctl.build_jobs(obs) for a in j.actions]
    assert 'HARVEST' not in before, 'Unknown deadline sentinel must not trigger expiry harvest'
    tile['max_lifespan_step']=264
    after=[a[0] for j in ctl.build_jobs(obs) for a in j.actions]
    assert 'HARVEST' in after
    assert ctl.animal_project('COW',11).output==21 and ctl.animal_project('SHEEP',11).output==22
    result={'status':'PASS','checks':['unknown_deadline_not_expired','known_deadline_generates_harvest','day29_cow_and_sheep_batches']}
    print(json.dumps(result))


if __name__=='__main__':main()
