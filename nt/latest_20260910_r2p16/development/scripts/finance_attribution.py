"""Observed Thomas branch response and daily paired state differences."""
from pathlib import Path
import gzip
import json
import importlib.util
import run_panel as panel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def loadtrace(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def routes(trace,module,seat):
    route=module.DEFAULT_ROUTE
    result=[]
    for frame in trace['days']:
        if frame['step']<719 and frame['step']%72==0:
            block=frame['step']//72
            route=module.route_for(block,frame['observations'][seat],route)
            result.append(dict(step=frame['step'],route=route))
    return result


def main():
    src=ROOT/'research/public_opponents_r2_20260910/thomas_955_v2/working/main.py'
    spec=importlib.util.spec_from_file_location('thomas_offline_attribution',src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    base={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    target=HERE/'finance_screen16'
    new=json.loads((target/'rows.json').read_text())
    report=[]
    for r in new:
        b=base[(r['opponent'],r['seed'],r['opponent_seat'])]
        if r['joint_action_sha256']==b['joint_action_sha256']:continue
        assert r['opponent']=='thomas_955_v2'
        before=loadtrace(ROOT/b['trace']);after=loadtrace(target/r['trace'])
        seat=r['opponent_seat'];own=1-seat
        days=[]
        for x,y in zip(before['days'],after['days']):
            assert x['step']==y['step']
            if x['step']<144:continue
            state=[]
            for frame in (x,y):
                obs=frame['observations'][own]
                state.append(dict(own_money=obs['farms'][own]['money'],opp_money=obs['farms'][seat]['money'],
                                  own=obs['farms'][own],debug=frame['r2_debug'],prices=obs['market']['prices']))
            days.append(dict(step=x['step'],before=state[0],after=state[1]))
        report.append(dict(seed=r['seed'],opponent_seat=seat,old_margin=b['r2_margin'],new_margin=r['r2_margin'],
                           old_shops=b['shops'],new_shops=r['shops'],shop_sequence_changed=b['shops']!=r['shops'],
                           before_routes=routes(before,mod,seat),after_routes=routes(after,mod,seat),days=days))
    panel.save(target/'FINANCE_ATTRIBUTION.json',dict(source_sha256=panel.sha(src),cases=report))
    print(json.dumps([dict(seed=r['seed'],seat=r['opponent_seat'],old=r['old_margin'],new=r['new_margin'],
                          old_routes=r['before_routes'],new_routes=r['after_routes']) for r in report]),flush=True)


if __name__=='__main__':main()
