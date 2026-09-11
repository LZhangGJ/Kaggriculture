"""Synthetic edge fixtures, explicitly separate from official full-game proof."""
import argparse
import collections
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time
import zlib
from check_lynn_native import EXP, PACKAGE, Reference, native, native_state, normalized, digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    asset=EXP/'native/lynn_v5_frozen.json.zlib';data=json.loads(zlib.decompress(asset.read_bytes()))
    for rel,h in data['source_hashes'].items():assert digest(PACKAGE/rel)==h
    opponent=native.LynnV5(data);params=json.loads((EXP/'profiles/candidates.json').read_text())['S3C03']
    env=native.Env(20261401);ctl=native.Controller(params);state=native.LynnState();snapshots={}
    for step in range(719):
        snapshots[step]=copy.deepcopy(env.observation(1))
        env.step([ctl.act(env,0),opponent.act(env,1,state)])
    rows=[];started=time.perf_counter()

    def check(name,observations):
        ref=Reference();st=native.LynnState();h=hashlib.sha256()
        for obs in observations:
            actual=opponent.probe_observation(obs,st);expected=ref.agent(copy.deepcopy(obs))
            cs,ps=native_state(st),ref.state(obs['player'])
            if normalized(actual)!=normalized(expected) or cs!=ps:
                (out/'failure.json').write_text(json.dumps(dict(name=name,observation=obs,actual=actual,expected=expected,native_state=cs,python_state=ps),indent=2))
                raise AssertionError((name,obs['step']))
            h.update(json.dumps(dict(action=actual,state=cs),sort_keys=True).encode())
        rows.append(dict(name=name,steps=len(observations),sha256=h.hexdigest(),assignments=cs['assignments'],
                         deferred=any(x['deferred'] for x in cs['decision_rows']),
                         delivery=[x['intervention_steps'] for x in cs['delivery']]))

    # Exact threshold +/- rounding and the information-maturity gate. Each
    # sequence keeps history across 88/150/169/176/313, including the cow cap.
    for shops,delta,(cows,sheep) in itertools.product(
            ([],['YARN_STORE'],['PET_CAFE'],['PIZZA_SHOP'],['YARN_STORE','BAKERY']),
            (-1,0,1,99,100,101,199,200,201,600),((0,0),(4,0),(0,4))):
        sequence=[]
        for step in (88,150,169,176,313):
            obs=copy.deepcopy(snapshots[step]);obs['town']['unlocked_shops']=shops[:]
            obs['market']['prices']['MILK']=160;obs['market']['prices']['WOOL']=max(1,200+delta)
            board=[[None for _ in range(10)] for _ in range(10)]
            for i in range(cows+sheep):board[0][i]=dict(kind='PASTURE',animal='COW' if i<cows else 'SHEEP')
            obs['farms'][0]['tiles']=board;sequence.append(obs)
        check(f'pressure:{shops}:{delta}:{cows}:{sheep}',sequence)

    for step,cash,stock in itertools.product((0,313,718),(0,22,1588,1589,1688,1689,100000),(0,90,91,100)):
        obs=copy.deepcopy(snapshots[step]);obs['farms'][1]['money']=cash
        obs['private']['shed']={k:0 for k in obs['private']['shed']};obs['private']['shed']['WHEAT']=stock
        check(f'funding:{step}:{cash}:{stock}',[obs])

    # Entry ordering, same-turn DROP/PLACE/PICKUP, terminal slot expansion,
    # below/above/floor price curves and both cargo dictionary orders.
    for step,iv,reverse in itertools.product((0,168,313,718),(0,9000,9999,10000,10001,11000,13000),(False,True)):
        obs=copy.deepcopy(snapshots[step]);own=obs['farms'][1];own['money']=100000
        own['farmer']=[4,4];own['hands']=[[4,4] for _ in own['hands']]
        products=list(obs['market']['inventory'])
        for item in products:obs['market']['inventory'][item]=iv
        obs['private']['shed']={k:(7 if k in products else 0) for k in obs['private']['shed']}
        items=[('WOOL',8),('MILK',8),('FERTILIZER',8),('WHEAT',8)]
        if reverse:items.reverse()
        obs['private']['inventories']=[dict(items) for _ in obs['private']['inventories']]
        check(f'ledger:{step}:{iv}:{reverse}',[obs])

    # Repeated-step reset is intentionally different in e773 (<) and the
    # other layers (<=). Backwards and seat-reuse must clear the right memory.
    sequence=[copy.deepcopy(snapshots[i]) for i in (0,88,150,150,169,313,314,315,316,718,0,313,314,88)]
    check('repeat_and_backwards',sequence)
    receipt=dict(status='PASS',build=json.loads((EXP/'native/build/build_receipt.json').read_text()),
                 asset_sha256=digest(asset),source_hashes=data['source_hashes'],cases=len(rows),
                 differential_calls=sum(x['steps'] for x in rows),mismatches=0,rows=rows,
                 deferred_cases=sum(x['deferred'] for x in rows),seconds=time.perf_counter()-started,
                 caveat='Synthetic public/own observations test branch equivalence; not naturally reached matches, not strength or exhaustive proof.')
    assert receipt['deferred_cases']>0
    (out/'acceptance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('build','source_hashes','rows')}),flush=True)


if __name__=='__main__':main()
