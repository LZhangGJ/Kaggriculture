from run100_common import *
from bias2_model import Bias2Model,update_aux
import torch,argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('control','aux'),required=True)
    parser.add_argument('--run',type=int,choices=(0,1),required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    assert read(P/'G0_ACCEPTANCE.json')['status']=='PASS';check_hashes();configure_torch()
    torch.manual_seed(701+args.run);m=Bias2Model().cuda();opt=torch.optim.Adam(m.parameters(),lr=3e-4)
    out=P/'training'/f'{args.arm}_r{args.run}';start_step=0;history=[];selection=[];elapsed_before=0.
    if out.exists():
        if not args.resume:raise FileExistsError(out)
        progress=read(out/'PROGRESS.json');history=progress['history'];start_step=len(history);elapsed_before=progress.get('elapsed',0.)
        selection=read(out/'selection.json')
        checkpoint=torch.load(out/f'step{start_step:03}.pt',weights_only=False)
        assert checkpoint['step']==start_step
        m.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        assert digest(out/f'step{start_step:03}.bin')==checkpoint['bin_sha']
    else:
        out.mkdir(parents=True,exist_ok=False);m.export(out/'step000.bin')
        assert digest(out/'step000.bin')==digest(OLD/'training'/f'f3_r{args.run}'/'step000.bin')
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=0,bin_sha=digest(out/'step000.bin')),out/'step000.pt')
    pool=runtime.make_pool();tic=time.perf_counter();weight=.1 if args.arm=='aux'else 0
    def dev(step):
        if any(s['step']==step for s in selection):return
        destination=out/f'dev_{step:03}'
        s=read(destination/'summary.json') if (destination/'summary.json').exists()else evaluate(pool,destination,out/f'step{step:03}.bin',1,69800000,32,9301)
        selection.append(dict(step=step,score=[s['overall']['win_rate'],s['overall']['mean_margin'],-step],summary=s))
        save(out/'selection.json',selection)
    dev(start_step if start_step%20==0 else 0)
    if not (out/'PROGRESS.json').exists():save(out/'PROGRESS.json',dict(status='RUNNING',history=history,elapsed=0.))
    for step in range(start_step+1,101):
        destination=out/f'rollout_{step:03}'
        if destination.exists():
            # An interrupted post-rollout/pre-update batch can reuse its on-policy data.
            with np.load(destination/'decisions.npz')as archive:r={k:archive[k]for k in archive.files}
            r['mode']=2;rs=read(destination/'summary.json')
            assert read(destination/'input.json')['sha']==digest(out/f'step{step-1:03}.bin')
        else:
            r=rollout(pool,out/f'step{step-1:03}.bin',2,69600000+args.run*100000+(step-1)*16,16,70000+args.run*10000+step)
            store_result(destination,r);save(destination/'input.json',dict(sha=digest(out/f'step{step-1:03}.bin'),keep_bonus=2.));rs=summary(r)
        torch.cuda.synchronize();t=time.perf_counter();stats=update_aux(m,opt,r,'cuda',step+args.run*1000,weight)
        torch.cuda.synchronize();dt=time.perf_counter()-t
        m.export(out/f'step{step:03}.bin')
        # Checkpoint and optimizer are written before advancing the durable progress pointer.
        temporary=out/f'step{step:03}.pt.writing'
        torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,bin_sha=digest(out/f'step{step:03}.bin')),temporary)
        temporary.replace(out/f'step{step:03}.pt')
        history.append(dict(step=step,training_games=step*224,rollout=rs,ppo=stats,update_seconds=dt,elapsed=elapsed_before+time.perf_counter()-tic))
        save(out/'PROGRESS.json',dict(status='RUNNING',history=history,elapsed=elapsed_before+time.perf_counter()-tic))
        print('TRAIN100',args.arm,args.run,step,'win',round(rs['overall']['win_rate'],4),'aux',round(stats['aux_loss'],5),
              'rollout_s',round(rs['call_seconds'],2),'update_s',round(dt,2),'KL',round(stats['approx_kl'],5),flush=True)
        if step%20==0:dev(step)
    best=max(selection,key=lambda s:tuple(s['score']));save(out/'best.json',best);check_hashes()
    save(out/'COMPLETE.json',dict(status='PASS',history=history,best=best,training_games=22400,keep_bonus=2.,aux_weight=weight,
                                  seconds=elapsed_before+time.perf_counter()-tic))
    save(out/'PROGRESS.json',dict(status='COMPLETE',history=history,elapsed=elapsed_before+time.perf_counter()-tic))
    print('TRAIN100_COMPLETE',args.arm,args.run,'best',best['step'],flush=True)

if __name__=='__main__':main()
