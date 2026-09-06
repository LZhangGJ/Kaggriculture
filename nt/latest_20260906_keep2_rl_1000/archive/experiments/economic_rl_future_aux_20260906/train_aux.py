from common import *
from aux_model import AuxModel, update_aux
import argparse, torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('control','aux'),required=True)
    parser.add_argument('--run',type=int,choices=(0,1),required=True);args=parser.parse_args()
    assert read(P/'G0_ACCEPTANCE.json')['status']=='PASS'
    protocol=read(P/'G2_PROTOCOL.json')
    for f,h in protocol['hashes'].items():assert digest(f)==h,f
    configure_torch();torch.manual_seed(701+args.run);m=AuxModel().cuda();opt=torch.optim.Adam(m.parameters(),lr=3e-4)
    out=P/'training'/f'{args.arm}_r{args.run}';out.mkdir(parents=True,exist_ok=False)
    m.export(out/'step000.bin');torch.save(m.state_dict(),out/'step000.pt')
    assert digest(out/'step000.bin')==digest(OLD/'training'/f'f3_r{args.run}'/'step000.bin')
    pool=runtime.make_pool();history=[];selection=[];tic=time.perf_counter();weight=.1 if args.arm=='aux' else 0
    def dev(step):
        s=evaluate(pool,out/f'dev_{step:03}',out/f'step{step:03}.bin',1,69200000,16,9301)
        selection.append(dict(step=step,score=[s['overall']['win_rate'],s['overall']['mean_margin'],-step],summary=s))
        save(out/'selection.json',selection)
    dev(0)
    for step in range(1,21):
        r=rollout(pool,out/f'step{step-1:03}.bin',2,69000000+args.run*100000+(step-1)*16,16,70000+args.run*10000+step)
        store_result(out/f'rollout_{step:03}',r)
        torch.cuda.synchronize();t=time.perf_counter()
        stats=update_aux(m,opt,r,'cuda',step+args.run*1000,weight)
        torch.cuda.synchronize();dt=time.perf_counter()-t
        m.export(out/f'step{step:03}.bin');torch.save(m.state_dict(),out/f'step{step:03}.pt')
        entry=dict(step=step,training_games=step*224,rollout=summary(r),ppo=stats,update_seconds=dt,elapsed=time.perf_counter()-tic)
        history.append(entry);save(out/'PROGRESS.json',dict(status='RUNNING',history=history))
        print('TRAIN',args.arm,args.run,step,'win',entry['rollout']['overall']['win_rate'],'aux',stats['aux_loss'],
              'rollout_s',round(r['wall_seconds'],2),'update_s',round(dt,2),flush=True)
        if step in (5,10,20):dev(step)
    best=max(selection,key=lambda s:tuple(s['score']));save(out/'best.json',best)
    for f,h in protocol['hashes'].items():assert digest(f)==h,f
    save(out/'COMPLETE.json',dict(status='PASS',history=history,best=best,training_games=4480,seconds=time.perf_counter()-tic,aux_weight=weight))
    save(out/'PROGRESS.json',dict(status='COMPLETE',history=history))
    print('TRAIN_DONE',args.arm,args.run,'best',best['step'],flush=True)

if __name__=='__main__':main()
