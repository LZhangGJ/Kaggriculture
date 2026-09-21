import copy,time,unittest
from kaggle_environments import make
from kaggle_environments.utils import structify
from ppo.stream_env import step,history_snapshot
from worker_phase import engine
from features_v2 import PublicHistory
from ppo.fast_features import encode_exact_fast
from ppo.test_fast import equal

class StreamChecks(unittest.TestCase):
    def test_invalid_actions(self):
        for actions in ([None,None],[7,{}],[{},[]],[7,{'farmer':['NORTH']}],[{'farmer':['SOUTH']},7]):
            a=make("kaggriculture",configuration=dict(seed=919323,episodeSteps=720),debug=False)
            b=make("kaggriculture",configuration=dict(seed=919323,episodeSteps=720),debug=False)
            a.reset(2);b.reset(2);a.step(actions);step(b,actions);equal(a.state,b.state)

    def test_full_games(self):
        timings=[0.,0.]
        for seed in (919321,919322):
            a=make('kaggriculture',configuration=dict(seed=seed,episodeSteps=720),debug=False);b=make('kaggriculture',configuration=dict(seed=seed,episodeSteps=720),debug=False)
            a.reset(2);b.reset(2);ha=[PublicHistory(),PublicHistory()];hb=[PublicHistory(),PublicHistory()]
            for turn in range(719):
                actions=[]
                for seat in (0,1):
                    oa=dict(a.state[seat].observation);ob=dict(b.state[seat].observation);oa['step']=ob['step']=turn
                    ha[seat].observe(oa);hb[seat].observe(ob)
                    if turn%53==0:
                        equal(encode_exact_fast(oa,ha[seat]),encode_exact_fast(ob,hb[seat]))
                    action=engine().agents['starter'](structify(oa))
                    if turn%37==0: action['market']=[['HIRE'],['HIRE'],['BUY_LAND'],['BUY_SEED','WHEAT',1000],['SELL','WHEAT',1000]]
                    actions.append(action)
                    # Exercise nonempty request history as well as observations.
                    ha[seat].requests=copy.deepcopy(action.get('market',[]));hb[seat].requests=copy.deepcopy(action.get('market',[]))
                    hb[seat].previous=history_snapshot(ob)
                t=time.perf_counter();a.step(copy.deepcopy(actions));timings[0]+=time.perf_counter()-t
                t=time.perf_counter();step(b,copy.deepcopy(actions));timings[1]+=time.perf_counter()-t
                equal(a.state,b.state)
            self.assertTrue(a.done and b.done)
        print('step_seconds official,stream',timings,flush=True)
if __name__=='__main__':unittest.main()
