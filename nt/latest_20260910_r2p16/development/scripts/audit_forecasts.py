"""Audit original R2's visible-state forecasts against its actual live games.

Recorded joint actions reproduce the measured baseline here, ONLY for labels.
Each own decision must match the recorded original action; the telemetry API
receives no future state. This is not new-strength testing or hindsight search.
"""
from pathlib import Path
import concurrent.futures as futures
import ctypes,gzip,json,multiprocessing,time
import run_panel as panel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'forecast_audit'
PROBE=HERE/'forecast_probe/policy/forecast_probe.so'


def one(row):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}"
    target=OUT/(key+'.json.gz')
    if target.exists():
        data=json.loads(gzip.decompress(target.read_bytes()))
        assert data['status']=='PASS' and data['source_action_hash']==row['joint_action_sha256']
        return dict(key=key,status='PASS',reused=True)
    trace=json.loads(gzip.decompress((ROOT/row['trace']).read_bytes()))
    saved={d['step']:d['observations'] for d in trace['days']}
    agent=panel.R2_MODULE.Agent(binary_path=str(PROBE))
    agent.lib.td_forecast_json.argtypes=[ctypes.c_void_p]
    agent.lib.td_forecast_json.restype=ctypes.c_char_p
    env=panel.old.LocalGame(row['seed'],panel.ENGINE)
    own=1-row['opponent_seat'];forecasts=[]
    try:
        for step,joint in enumerate(trace['actions']):
            observation=env.observation(own)
            if step in saved:assert observation==saved[step][own]
            action=agent(observation)
            assert action==joint[own],dict(key=key,step=step,original=joint[own],probe=action)
            if step%24==0:
                data=json.loads(agent.lib.td_forecast_json(agent.handle).decode())
                data.update(step=step,shops=observation['town']['unlocked_shops'],
                            market=observation['market'],own_money=observation['farms'][own]['money'],
                            opponent_money=observation['farms'][1-own]['money'])
                forecasts.append(data)
            env.advance(joint)
        assert env.done and env.t==719 and env.observation(own)==saved[719][own]
        target.parent.mkdir(parents=True,exist_ok=True)
        data=dict(status='PASS',row=row,steps=719,source_action_hash=row['joint_action_sha256'],forecasts=forecasts)
        target.write_bytes(gzip.compress(json.dumps(data,separators=(',',':')).encode(),compresslevel=1))
        return dict(key=key,status='PASS',reused=False)
    finally:agent.close()


def main():
    selection=json.loads((HERE/'baseline_audit/SELECTION.json').read_text())['rows']
    OUT.mkdir(exist_ok=True)
    protocol=dict(selection='baseline_audit/SELECTION.json',cases=len(selection),
                  probe_sha256=panel.sha(PROBE),release_sha256=panel.sha(panel.old.R2/'agent.so'),
                  config_sha256=panel.sha(panel.old.R2/'config.json'),engine_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),
                  inputs='Current seat observation only; telemetry after native decision; future used offline only')
    path=OUT/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:panel.save(path,protocol)
    started=time.perf_counter();results=[]
    with futures.ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=panel.init_worker) as executor:
        tasks=[executor.submit(one,row) for row in selection]
        for f in futures.as_completed(tasks):
            results.append(f.result())
            if len(results)%40==0 or len(results)==len(selection):
                progress=dict(done=len(results),total=len(selection),seconds=time.perf_counter()-started)
                panel.save(OUT/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    panel.save(OUT/'RESULTS.json',dict(status='PASS',cases=len(results),actions=719*len(results),
                                    seconds=time.perf_counter()-started,rows=results))


if __name__=='__main__':main()
