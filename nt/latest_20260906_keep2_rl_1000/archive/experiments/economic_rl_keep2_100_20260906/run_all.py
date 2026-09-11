from run100_common import *
from bias2_model import Bias2Model,forecast_metrics
import subprocess,torch

def main():
    assert read(P/'G0_ACCEPTANCE.json')['status']=='PASS'
    if not (P/'PROTOCOL.json').exists():
        receipt=read(P/'BUILD_RECEIPT.json');hashes=dict(receipt['preserved']);hashes.update(receipt['built'])
        files=[*P.glob('*.py'),P/'policy.cpp',P/'PLAN_ZH.md',AUX/'aux_model.py',AUX/'common.py',OLD/'model.py',OLD/'runtime.py',OLD/'rl_common.hpp']
        hashes.update({str(f):digest(f)for f in files})
        for f,h in hashes.items():assert digest(f)==h,f
        save(P/'PROTOCOL.json',dict(hashes=hashes,keep_bonus=2.,rounds=100,repeats=2,arms=['control','aux'],games_per_round=224,
             seed_starts=dict(training=[69600000,69700000],development=69800000,final=69900000),
             development_seeds=32,final_seeds=100,checkpoint_options=[0,20,40,60,80,100],
             selection='development greedy win, margin, earlier checkpoint; all frozen before final'))
    check_hashes();(P/'logs').mkdir(exist_ok=True);tic=time.perf_counter()
    for rep in (0,1):
        for arm in ('control','aux'):
            directory=P/'training'/f'{arm}_r{rep}'
            if not (directory/'COMPLETE.json').exists():
                command=[sys.executable,str(P/'train100.py'),'--arm',arm,'--run',str(rep)]
                if directory.exists():command.append('--resume')
                with (P/'logs'/f'{arm}_r{rep}.log').open('a')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            print('FINISHED',arm,rep,read(directory/'best.json')['step'],flush=True)
    chosen={}
    for rep in (0,1):
        for arm in ('control','aux'):
            name=f'{arm}_r{rep}';directory=P/'training'/name;best=read(directory/'best.json')['step']
            chosen[name]=dict(best_step=best,selected=str(directory/f'step{best:03}.bin'),
                step20=str(directory/'step020.bin'),step100=str(directory/'step100.bin'),
                hashes={str(f):digest(f)for f in (directory/f'step{best:03}.bin',directory/'step020.bin',directory/'step100.bin')})
    if (P/'FINAL_SELECTION_FROZEN.json').exists():assert read(P/'FINAL_SELECTION_FROZEN.json')==chosen
    else:save(P/'FINAL_SELECTION_FROZEN.json',chosen)
    (P/'TRAINING_ACCEPTANCE_ZH.md').write_text('# KEEP=2 100轮训练完成\n\n四次运行均完成100轮，每次22,400完整局，共89,600训练局。第0/20/40/60/80/100轮检查固定开发集；最终选定权重已冻结，此后才启动最终新种子测试。完整模型和优化器均保存。训练完成不代表强度通过。\n',encoding='utf8')
    configure_torch();pool=runtime.make_pool();base=P/'final';base.mkdir(exist_ok=True)
    if not (base/'keep/summary.json').exists():evaluate(pool,base/'keep')
    forecasts={}
    for name,config in chosen.items():
        runs=[('selected',config['selected'],1),('step20_greedy',config['step20'],1),('step20_sample',config['step20'],2),
              ('step100_greedy',config['step100'],1),('step100_sample',config['step100'],2)]
        for label,checkpoint,mode in runs:
            dest=base/f'{name}_{label}'
            if not (dest/'summary.json').exists():evaluate(pool,dest,checkpoint,mode)
            if label=='step100_sample':
                m=Bias2Model().cuda();m.load_state_dict(torch.load(Path(checkpoint).with_suffix('.pt'),weights_only=False)['model'])
                with np.load(dest/'decisions.npz')as archive:data={k:archive[k]for k in ('global','candidates','choice','game','day')}
                forecasts[name]=forecast_metrics(m,data)
    save(P/'FORECAST_RESULTS.json',forecasts);check_hashes()
    save(P/'RUN_COMPLETE.json',dict(status='PASS',training_games=89600,final_games=29400,seconds=time.perf_counter()-tic,
                                   old_inputs_unchanged=True,all_selection_before_final=True))
    subprocess.run([sys.executable,str(P/'report100.py')],check=True)
    print('ALL_DONE',flush=True)

if __name__=='__main__':main()
