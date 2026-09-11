from common900 import *
import argparse,shutil


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['control','aux'],required=True)
    ap.add_argument('--run',type=int,choices=[0,1],required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    assert read(P/'G0_ACCEPTANCE.json')['status']=='PASS';check_hashes();configure_torch()
    name=f'{args.arm}_r{args.run}';source=SOURCE/'training'/name;out=P/'training'/name
    if out.exists():
        assert args.resume
        progress=read(out/'PROGRESS.json');history=progress['history'];step0=800+len(history)
        m,opt,state=restore(out/f'step{step0:03}.pt');selection=read(out/'selection.json')
        elapsed_before=progress['elapsed']
    else:
        out.mkdir(parents=True,exist_ok=False)
        m,opt,state=restore(source/'step800.pt');assert state['step']==800
        for ext in ('bin','pt'):shutil.copyfile(source/f'step800.{ext}',out/f'step800.{ext}')
        history=[];step0=800;elapsed_before=0.
        selection=read(source/'selection.json');save(out/'selection.json',selection)
        save(out/'INITIAL_STATE.json',dict(source=str(source/'step800.pt'),sha=digest(source/'step800.pt'),
            bin_sha=digest(source/'step800.bin'),optimizer=optimizer_receipt(opt)))
        save(out/'PROGRESS.json',dict(status='RUNNING',history=[],elapsed=0.))
    assert state['step']==step0
    assert optimizer_receipt(opt)['steps']==[step0*56]
    pool=runtime.make_pool();tic=time.perf_counter();weight=.1 if args.arm=='aux'else 0.
    def dev(step):
        if step<=800 or any(s['step']==step for s in selection):return
        dest=out/f'dev_{step:03}'
        s=read(dest/'summary.json')if (dest/'summary.json').exists()else evaluate(pool,dest,out/f'step{step:03}.bin',1,69800000,32,9301)
        selection.append(dict(step=step,score=[s['overall']['win_rate'],s['overall']['mean_margin'],-step],summary=s))
        save(out/'selection.json',selection)
    if step0%20==0:dev(step0)
    for step in range(step0+1,901):
        folder=out/f'rollout_{step:03}';previous=out/f'step{step-1:03}.bin'
        if folder.exists():
            with np.load(folder/'decisions.npz')as z:r={k:z[k]for k in z.files}
            r['mode']=2;rs=read(folder/'summary.json')
            assert read(folder/'input.json')['sha']==digest(previous)
        else:
            r=rollout(pool,previous,2,69600000+args.run*100000+(step-1)*16,16,70000+args.run*10000+step)
            store_result(folder,r);save(folder/'input.json',dict(sha=digest(previous),keep_bonus=2.));rs=summary(r)
        torch.cuda.synchronize();tt=time.perf_counter()
        stats=update_aux(m,opt,r,'cuda',step+args.run*1000,weight)
        torch.cuda.synchronize();update_seconds=time.perf_counter()-tt
        checkpoint(out/f'step{step:03}.pt',m,opt,step)
        history.append(dict(step=step,new_training_games=(step-800)*224,total_training_games=step*224,
            rollout=rs,ppo=stats,update_seconds=update_seconds,elapsed=elapsed_before+time.perf_counter()-tic))
        save(out/'PROGRESS.json',dict(status='RUNNING',history=history,elapsed=elapsed_before+time.perf_counter()-tic))
        print('TRAIN900',name,step,'win',round(rs['overall']['win_rate'],4),'rollout_s',round(rs['call_seconds'],2),
              'update_s',round(update_seconds,2),'kl',round(stats['approx_kl'],5),flush=True)
        if step%20==0:dev(step)
    best=max(selection,key=lambda s:tuple(s['score']));save(out/'best.json',best)
    final_optimizer=optimizer_receipt(opt);assert final_optimizer['steps']==[50400]
    check_hashes();elapsed=elapsed_before+time.perf_counter()-tic
    save(out/'COMPLETE.json',dict(status='PASS',history=history,best=best,new_training_games=22400,total_training_games=201600,
        keep_bonus=2.,aux_weight=weight,seconds=elapsed,final_optimizer=final_optimizer))
    save(out/'PROGRESS.json',dict(status='COMPLETE',history=history,elapsed=elapsed))
    print('TRAIN900_COMPLETE',name,'best',best['step'],flush=True)


if __name__=='__main__':main()
