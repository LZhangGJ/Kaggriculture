"""Frozen official-trajectory regression, NOT a new strength evaluation."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'referee'))
from cpu_runtime import LocalGame,load_engine
from main import create_agent
FIELDS=['farms','private','market','town','day','hour','player']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(binary,index_file=None):
    binary=Path(binary).resolve(strict=True);index=json.loads((index_file or (ROOT/'tests/GOLDEN.json')).read_text())
    if not index:raise RuntimeError('Missing frozen action references')
    result={'status':'PASS','scope':'Recorded-opponent behavior regression, not live strength evidence','binary_sha256':sha(binary),'agent_action_comparisons':0,'fixture_action_comparisons':0,'state_checkpoints':0,'cases':[]}
    prefix=json.loads((ROOT/'fixtures/melon12_opening.json').read_text())['actions']
    for item in index:
        path=ROOT/item['file']
        if sha(path)!=item['sha256']:raise RuntimeError('Frozen trajectory checksum mismatch')
        if item['config_sha256']!=sha(ROOT/'policy/config.json'):raise RuntimeError('Configuration differs from reference')
        data=json.loads(gzip.decompress(path.read_bytes()));seat=data['result']['opponent_seat'];own=1-seat
        env=LocalGame(data['result']['seed'],load_engine());player=create_agent(binary);generated=used_prefix=0
        frames={d['step']:d['observations']for d in data['days']}
        try:
            if len(data['actions'])!=719:raise RuntimeError('Incomplete reference')
            for step,pair in enumerate(data['actions']):
                if step in frames:
                    for s in (0,1):
                        observed=env.observation(s)
                        for field in FIELDS:
                            if observed[field]!=frames[step][s][field]:raise RuntimeError(f'{path.name}: state step={step} seat={s} field={field}')
                    result['state_checkpoints']+=1
                if env.configuration.seed is not None:raise RuntimeError('Seed leaked into public configuration')
                obs=env.observation(own)
                if 'seed'in obs:raise RuntimeError('Seed leaked into policy observation')
                if item['opening']=='m12'and step<24:action=copy.deepcopy(prefix[step]);used_prefix+=1
                else:action=player(obs);generated+=1
                if action!=pair[own]:raise RuntimeError(f'{path.name}: action mismatch step={step}: got={action} expected={pair[own]}')
                actual=copy.deepcopy(pair);actual[own]=action;env.advance(actual)
            if not env.done or env.t!=719:raise RuntimeError('Referee did not terminate')
            cash=env.observation(own)['farms']
            if cash[own]['money']!=data['result']['own_cash']or cash[seat]['money']!=data['result']['opponent_cash']:raise RuntimeError('Final cash mismatch')
            result['agent_action_comparisons']+=generated;result['fixture_action_comparisons']+=used_prefix
            result['cases'].append({'file':item['file'],'generated':generated,'prefix':used_prefix,'steps':env.t})
        finally:player.close()
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,default=ROOT/'policy/joint.so');p.add_argument('--out',type=Path);p.add_argument('--index',type=Path);a=p.parse_args()
    try:r=check(a.binary,a.index)
    except Exception as e:r={'status':'FAIL','error':str(e)}
    if a.out:
        if a.out.exists():raise SystemExit('Refusing overwrite')
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
if __name__=='__main__':main()
