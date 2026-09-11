"""Can public tile changes identify finite-crop turnover without private data?"""
from pathlib import Path
from collections import Counter,defaultdict
import copy,gzip,json,time,concurrent.futures as futures,multiprocessing
import run_panel as panel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'public_crop_removal_audit_v2'
FIRST={'WHEAT':2,'CARROT':2,'MELON':10}

def infer(old,new,step):
    events=[]
    for y in range(10):
        for x in range(10):
            a=old['tiles'][y][x];b=new['tiles'][y][x]
            if not isinstance(a,dict) or a.get('kind')!='PLANT' or a.get('crop') not in FIRST:continue
            crop=a['crop'];age=step//24-a['planted_day']
            if a['yield_units']<=0 or age<FIRST[crop]:continue
            if a['max_lifespan_step']>=0 and a['max_lifespan_step']<=step+1:continue
            # End-of-day death without watering is not a harvested batch.
            if (step+1)%24==0 and not a['watered_today'] and a['consecutive_unwatered']>=1:continue
            # Only empty/replanted soil; never interpret WEED/BUILD as harvest.
            # Raw official schema uses None, not {kind: EMPTY} (native enum).
            gone=b is None or (isinstance(b,dict) and b.get('kind')=='PLANT' and
                   (b.get('crop')!=crop or b.get('planted_day')!=a['planted_day']))
            if gone:events.append(dict(pos=[x,y],crop=crop,age=age,step=step))
    return events

def one(row):
    path=OUT/f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}.json"
    if path.exists():return json.loads(path.read_text())
    trace=json.loads(gzip.decompress((ROOT/row['trace']).read_bytes()));saved={x['step']:x['observations'] for x in trace['days']};seat=row['opponent_seat']
    game=panel.old.LocalGame(row['seed'],panel.ENGINE);original=panel.ENGINE._apply_unit_action;labels=[];inferred=[]
    def unit(farm,private,idx,action,*args):
        is_op=farm is game.state[0].observation.farms[seat]
        pos=panel.ENGINE._farmer_position(farm,idx);tile=copy.deepcopy(farm['tiles'][pos[1]][pos[0]]) if is_op and pos else None
        ret=original(farm,private,idx,action,*args)
        if tile and action and action[0]=='HARVEST' and tile.get('crop') in FIRST:
            after=farm['tiles'][pos[1]][pos[0]]
            if after!=tile:labels.append(dict(step=game.t,pos=list(pos),crop=tile['crop'],age=game.t//24-tile['planted_day']))
        return ret
    panel.ENGINE._apply_unit_action=unit
    try:
        for step,joint in enumerate(trace['actions']):
            obs=game.observation(1-seat)
            if step in saved:assert obs==saved[step][1-seat]
            old=obs['farms'][seat];game.advance(joint)
            new=game.observation(1-seat)['farms'][seat]
            inferred.extend(infer(old,new,step))
        assert game.done and game.observation(seat)==saved[719][seat]
    finally:panel.ENGINE._apply_unit_action=original
    def key(e):return e['step'],tuple(e['pos']),e['crop'],e['age']
    labelkeys={key(x) for x in labels};detectedkeys={key(x) for x in inferred}
    data=dict(row=row,inferred=inferred,labels=labels,matched=sum(key(x) in labelkeys for x in inferred),false=[x for x in inferred if key(x) not in labelkeys],missed=[x for x in labels if key(x) not in detectedkeys])
    path.write_text(json.dumps(data,indent=2));return data

def main():
    OUT.mkdir(exist_ok=True);rows=json.loads((HERE/'proposal_equivalence_audit/SELECTION.json').read_text());started=time.perf_counter()
    with futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as pool:cases=list(pool.map(one,rows))
    age=defaultdict(Counter)
    for c in cases:
        for e in c['inferred']:age[e['crop']][e['age']]+=1
    result=dict(cases=len(cases),steps=719*len(cases),inferred=sum(len(c['inferred']) for c in cases),matched=sum(c['matched'] for c in cases),false=sum(len(c['false']) for c in cases),missed=sum(len(c['missed']) for c in cases),age_histograms=dict(age),seconds=time.perf_counter()-started,
                boundary='Labels from referee only for offline validation. Live inference accepts two successive public farms and step, not actual actions/private/seed. DIG may remain observationally ambiguous on another policy.')
    panel.save(OUT/'RESULTS.json',result)
    (OUT/'REPORT_ZH.md').write_text('# 公开有限作物腾地节奏识别\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
