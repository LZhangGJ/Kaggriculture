"""Check a native rebuild control and the untouched first 24 days."""
from pathlib import Path
import contextlib
import copy
import io
import json
import sys

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification'))
from policy_host import Policy,LocalGame,load_engine


def main():
    pool=json.loads((HERE/'SOURCE_POOL.json').read_text())['opponents']
    seeds=json.loads((HERE/'SEEDS.json').read_text())['development']
    receipts=[]
    for index,opid in enumerate(('external/n14_prvsiyan','internal/r14_cashflow')):
        rival=next(r for r in pool if r['id']==opid)
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            old=Policy(str(HERE/'candidates/cf_liq_nointraday/main.py'),'parent')
            enabled=Policy(str(HERE/'candidates/cf_liq_terminal6/main.py'),'enabled')
            factory=enabled.fn.__globals__['create_agent'];control=factory()
            control.inner.config['r14_terminal_mode']=0;control.inner.reset()
            enemy=Policy(rival['entry'],'enemy');env=LocalGame(seeds[index],load_engine())
            full=prefix=0; mismatches=[]
            while not env.done:
                obs=env.observation(0);act=old(obs,copy.deepcopy(env.configuration))
                copyact=control(obs,copy.deepcopy(env.configuration))
                if act==copyact:full+=1
                else:mismatches.append(dict(step=env.t,kind='disabled',parent=act,new=copyact))
                if env.t<576:
                    newact=enabled(obs,copy.deepcopy(env.configuration))
                    if act==newact:prefix+=1
                    else:mismatches.append(dict(step=env.t,kind='prefix',parent=act,new=newact))
                enemyact=enemy(env.observation(1),copy.deepcopy(env.configuration));env.advance([act,enemyact])
            control.close()
        receipts.append(dict(opponent=opid,seed=seeds[index],full_matches=full,full_expected=719,
                             prefix_matches=prefix,prefix_expected=576,mismatches=mismatches[:5]))
    output=dict(passed=all(r['full_matches']==719 and r['prefix_matches']==576 for r in receipts),rows=receipts)
    (HERE/'TERMINAL_PARITY.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output));assert output['passed']


if __name__=='__main__':main()
