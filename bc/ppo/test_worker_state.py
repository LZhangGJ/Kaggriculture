import copy,os,unittest
import numpy as np
import torch
from kaggle_environments import make
from bc_runtime import ITEMS
from exact_actions import worker_wire
from worker_phase import engine,resolve_worker_phase
from ppo.worker_state import WorkerState,WorkerScenarios,pack
from ppo.worker_features_device import WorkerFeatures
from exact_actions import worker_features,needs_quantity
from ppo.fast_features import worker_stats


class WorkerStateChecks(unittest.TestCase):
    def test_all_operations_and_prefix_cancellation(self):
        torch.set_num_threads(1)
        env=make('kaggriculture',configuration={'seed':919734,'episodeSteps':720},debug=False)
        env.reset(2);obs=copy.deepcopy(dict(env.state[0].observation))
        obs['day']=10;obs['hour']=3;obs['step']=243
        farm=obs['farms'][0];pr=obs['private'];farm['hands']=[[0,0],[0,0]]
        pr['inventories']=[dict(reversed([(i,5) for i in ITEMS])) for _ in range(3)]
        pr['shed']={'WHEAT':97};pr['seeds']={i:1 for i in ITEMS[:5]}
        plant=engine()._new_plant('WHEAT',1,24)
        plant.update(yield_units=4,watered_today=False)
        animal=engine()._new_animal('GOOSE',1)
        animal.update(yield_units=3,fertilizer_available=True)
        cases=[None,'LOCKED',{'kind':'WEED'},{'kind':'COOP'},{'kind':'PASTURE'},plant,animal]
        device=os.environ.get('PARITY_DEVICE','cpu')
        for tile in cases:
            for adjacent in (False,True):
                pos=list(engine()._shed_access_tiles(10)[0]) if adjacent else [0,0]
                farm['farmer']=pos;farm['hands']=[pos[:],pos[:]]
                farm['tiles'][pos[1]][pos[0]]=copy.deepcopy(tile)
                seed={k:torch.as_tensor(v[None],device=device) for k,v in pack(obs).items()}
                state=WorkerState(seed)
                scenarios=WorkerScenarios(seed)
                features=WorkerFeatures(state)
                for action in range(44):
                    # Same crop can be planted then cancelled by the next request.
                    ids=[action,15,15];quantities=[100,0,0]
                    wires=[worker_wire(i,q if i>=20 else None) for i,q in zip(ids,quantities)]
                    for length in (1,3):
                        state.resolve([torch.tensor([i],device=device) for i in ids[:length]],
                                      [torch.tensor([q],device=device) for q in quantities[:length]])
                        resolved=resolve_worker_phase(obs,wires[:length])[0]
                        reference=pack(resolved)
                        for key in ('tiles','positions','inventory','shed','seeds'):
                            np.testing.assert_array_equal(getattr(state,key)[0].cpu().numpy(),reference[key],err_msg=f'{tile} {adjacent} {action} {length} {key}')
                        other=scenarios.resolve([torch.tensor([i],device=device) for i in ids[:length]],
                                                [torch.tensor([q],device=device) for q in quantities[:length]])
                        for key in ('tiles','positions','inventory','shed','seeds','order'):
                            np.testing.assert_array_equal(getattr(other,key)[0].cpu().numpy(),getattr(state,key)[0].cpu().numpy())
                        got=state.order[0].cpu().numpy();expected=reference['order']
                        for a,b in zip(got,expected):
                            np.testing.assert_array_equal(np.argsort(a)[np.sort(a)>=0],np.argsort(b)[np.sort(b)>=0])
                        counts=torch.tensor([[sum(i==15+c for i in ids[:length]) for c in range(5)]],device=device)
                        compact,need,stats=features(0,counts)
                        expected=worker_features(obs,resolved,wires[:length],0)
                        for key,value in compact.items():
                            np.testing.assert_allclose(value[0].cpu().numpy(),expected[key],rtol=1e-7,atol=1e-7,err_msg=f'{tile} {adjacent} {action} {length} {key}')
                        np.testing.assert_array_equal(need[0].cpu().numpy(),[needs_quantity(resolved,0,i) for i in range(44)])
                        np.testing.assert_array_equal(stats[0].cpu().numpy(),worker_stats(resolved,0))

class AllCancellationMasks(unittest.TestCase):
    def test_every_crop_mask_and_prefix(self):
        torch.set_num_threads(1)
        env=make('kaggriculture',configuration={'seed':919734,'episodeSteps':720},debug=False)
        env.reset(2);base=copy.deepcopy(dict(env.state[0].observation))
        base['farms'][0]['hands']=[[i%5,i//5] for i in range(1,10)]
        base['farms'][0]['farmer']=[0,0]
        base['private']['inventories']=[{} for _ in range(10)]
        for y in range(2):
            for x in range(5):base['farms'][0]['tiles'][y][x]=None
        observations=[]
        for mask in range(32):
            obs=copy.deepcopy(base)
            obs['private']['seeds']={crop:1 if mask&(1<<i) else 2 for i,crop in enumerate(ITEMS[:5])}
            observations.append(obs)
        device=os.environ.get('PARITY_DEVICE','cpu')
        packed=[pack(o) for o in observations]
        seed={k:torch.as_tensor(np.stack([r[k] for r in packed]),device=device) for k in packed[0]}
        scenarios=WorkerScenarios(seed);features=WorkerFeatures(scenarios.state)
        if os.environ.get('PPO_COMPILE_WORKERS')=='1':
            features=torch.compile(features,fullgraph=True,dynamic=True,options={'triton.cudagraphs':False})
        prefix=[]
        for depth in range(10):
            action=15+depth%5
            scenarios.append(depth,torch.full((32,),action,device=device),torch.zeros(32,device=device,dtype=torch.long))
            prefix.append(worker_wire(action,None))
            expected=[resolve_worker_phase(o,prefix)[0] for o in observations]
            for key in ('tiles','positions','inventory','shed','seeds'):
                np.testing.assert_array_equal(getattr(scenarios.state,key).cpu().numpy(),np.stack([pack(o)[key] for o in expected]))
            if depth<9:
                compact,_,_=features(scenarios.state.workers[depth+1].expand(32),scenarios.counts)
                for row,(original,resolved) in enumerate(zip(observations,expected)):
                    reference=worker_features(original,resolved,prefix,depth+1)
                    for key,value in compact.items():
                        np.testing.assert_allclose(value[row].cpu().numpy(),reference[key],rtol=1e-7,atol=1e-7)

if __name__=='__main__':unittest.main()
