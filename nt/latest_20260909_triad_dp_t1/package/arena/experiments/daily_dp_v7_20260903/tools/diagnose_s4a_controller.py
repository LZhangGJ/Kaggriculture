"""Small live-controller diagnostics.  Never selects a policy or queries future state."""
from pathlib import Path
import argparse,json,sys

EXP=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP/'native/build'))
import _dp7_native as native

def pass_action(obs,seat):
    return {'farmer':['PASS'],'hands':[['PASS'] for _ in obs['farms'][seat]['hands']],'market':[]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--configs',required=True);p.add_argument('--labels',required=True)
    p.add_argument('--seed',type=int,default=20262701);p.add_argument('--seat',type=int,default=0);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);configs=json.loads(Path(a.configs).read_text());rows=[]
    for label in a.labels.split(','):
        env=native.Env(a.seed);ctl=native.Controller(configs[label]);daily=[]
        while not env.done:
            observations=[env.observation(0),env.observation(1)];step=observations[0]['step'];actions=[pass_action(observations[0],0),pass_action(observations[1],1)];actions[a.seat]=ctl.act(env,a.seat);env.step(actions)
            if step%24==23 or env.done:
                now=env.observation(a.seat);daily.append({'day':step//24,'cash':now['farms'][a.seat]['money'],'debug':ctl.debug()})
        rows.append({'label':label,'cash':env.observation(a.seat)['farms'][a.seat]['money'],'debug':ctl.debug(),'daily':daily})
    (out/'diagnostics.json').write_text(json.dumps({'status':'DIAGNOSTIC_ONLY','seed':a.seed,'seat':a.seat,'rows':rows},indent=2),encoding='utf8')
    for row in rows:print(json.dumps({k:row[k] for k in ('label','cash','debug')}))

if __name__=='__main__':main()
