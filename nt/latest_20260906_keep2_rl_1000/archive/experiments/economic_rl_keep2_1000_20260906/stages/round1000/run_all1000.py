from common1000 import *
import subprocess


def main():
    assert read(P/'G0_ACCEPTANCE.json')['status']=='PASS'
    if not (P/'PROTOCOL.json').exists():
        hashes=dict(read(SOURCE/'PROTOCOL.json')['hashes'])
        files=[*ROOT.glob('*.py'),ROOT/'PLAN_ZH.md',*P.glob('*.py'),P/'PLAN_ZH.md',P/'G0_ACCEPTANCE.json',SOURCE/'MANIFEST.json',SOURCE/'FINAL_AUDIT.json',SOURCE/'audit900.py',SOURCE/'preflight900.py']
        for name in ('control_r0','aux_r0','control_r1','aux_r1'):
            directory=SOURCE/'training'/name
            files.extend([directory/'step900.pt',directory/'step900.bin',directory/'selection.json',directory/'COMPLETE.json'])
            files.extend(directory/f'step{s:03}.bin'for s in (820,840,860,880))
        hashes.update({str(f):digest(f)for f in files})
        for f,h in hashes.items():assert digest(f)==h,f
        save(P/'PROTOCOL.json',dict(hashes=hashes,start_round=900,end_round=1000,keep_bonus=2.,repeats=2,
           arms=['control','aux'],games_per_round=224,new_training_seed_starts=[69614400,69714400],
           development_seed_start=69800000,development_seeds=32,final_seed_start=70300000,final_seeds=100,
           selection='same development greedy win, margin, earlier checkpoint; all historical dev candidates retained',
           initialization='each own step900 model AND optimizer, not best checkpoint'))
    check_hashes();(P/'logs').mkdir(exist_ok=True)
    started=P/'STARTED.json'
    if not started.exists():save(started,dict(unix_time=time.time()))
    for rep in (0,1):
        for arm in ('control','aux'):
            name=f'{arm}_r{rep}';directory=P/'training'/name
            if not (directory/'COMPLETE.json').exists():
                command=[sys.executable,str(P/'train1000.py'),'--arm',arm,'--run',str(rep)]
                if directory.exists():command.append('--resume')
                with (P/'logs'/f'{name}.log').open('a')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('FINISHED',name,'best',read(directory/'best.json')['step'],flush=True)
    chosen={}
    for rep in (0,1):
        for arm in ('control','aux'):
            name=f'{arm}_r{rep}';directory=P/'training'/name;best=read(directory/'best.json')['step']
            paths=dict(selected=chosen_path(name,best),step900=SOURCE/'training'/name/'step900.bin',step1000=directory/'step1000.bin')
            chosen[name]=dict(best_step=best,**{k:str(v)for k,v in paths.items()},hashes={str(v):digest(v)for v in paths.values()})
    if (P/'FINAL_SELECTION_FROZEN.json').exists():assert read(P/'FINAL_SELECTION_FROZEN.json')==chosen
    else:save(P/'FINAL_SELECTION_FROZEN.json',chosen)
    save(P/'TRAINING_ACCEPTANCE.json',dict(status='PASS',new_training_games=89600,total_training_games=896000,
        continued_from_step900_optimizer=True,final_selection_frozen=True))
    configure_torch();pool=runtime.make_pool();base=P/'final';base.mkdir(exist_ok=True)
    if not (base/'keep/summary.json').exists():evaluate(pool,base/'keep')
    forecasts={}
    for name,c in chosen.items():
        configs=[('selected',c['selected'],1),('step900_greedy',c['step900'],1),('step900_sample',c['step900'],2),
                 ('step1000_greedy',c['step1000'],1),('step1000_sample',c['step1000'],2)]
        for label,model,mode in configs:
            dest=base/f'{name}_{label}'
            if not (dest/'summary.json').exists():evaluate(pool,dest,model,mode)
            if label in ('step900_sample','step1000_sample'):
                m=Bias2Model().cuda();m.load_state_dict(torch.load(Path(model).with_suffix('.pt'),weights_only=False)['model'])
                with np.load(dest/'decisions.npz')as z:data={k:z[k]for k in ('global','candidates','choice','game','day')}
                forecasts[f'{name}_{label}']=forecast_metrics(m,data)
    save(P/'FORECAST_RESULTS.json',forecasts);check_hashes()
    if not (P/'RUN_COMPLETE.json').exists():
        save(P/'RUN_COMPLETE.json',dict(status='PASS',new_training_games=89600,total_training_games=896000,
            final_games=29400,seconds=time.time()-read(started)['unix_time'],all_selection_before_final=True,old_inputs_unchanged=True))
    subprocess.run([sys.executable,str(P/'report1000.py')],check=True)
    print('ALL_DONE',flush=True)


if __name__=='__main__':main()
