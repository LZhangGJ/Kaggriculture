"""Sequential matched-budget runs; freeze all model choices before final seeds."""
from common import *
from aux_model import AuxModel, forecast_metrics
import subprocess, torch

def main():
    assert read(P/'G0_ACCEPTANCE.json')['status']=='PASS'
    protocol_path=P/'G2_PROTOCOL.json'
    if not protocol_path.exists():
        paths=[*P.glob('*.py'),P/'PLAN_ZH.md',FIX/'build/f3.so',OLD/'model.py',OLD/'runtime.py',OLD/'rl_common.hpp']
        hashes={str(f):digest(f)for f in paths}
        hashes.update(read(OLD/'SOURCE_FREEZE.json'))
        fx=read(FIX/'ACCEPTANCE.json');hashes.update(fx['fixed_hashes'])
        for f,h in hashes.items():assert digest(f)==h,f
        save(protocol_path,dict(hashes=hashes,batches=20,games_per_batch=224,repeats=2,arms=['control','aux'],
             main_comparison='development-selected greedy, aux minus control; all final seeds unopened until freeze',
             final_seed_start=69300000,final_seed_count=100,checkpoint_selection_days=[0,5,10,20],aux_weight=.1,
             training_seed_starts=[69000000,69100000],dev_seed_start=69200000,dev_seed_count=16))
    protocol=read(protocol_path)
    (P/'logs').mkdir(exist_ok=True);tic=time.perf_counter()
    for rep in (0,1):
        for arm in ('control','aux'):
            directory=P/'training'/f'{arm}_r{rep}'
            if not (directory/'COMPLETE.json').exists():
                with (P/'logs'/f'{arm}_r{rep}.log').open('w')as log:
                    subprocess.run([sys.executable,str(P/'train_aux.py'),'--arm',arm,'--run',str(rep)],
                                   stdout=log,stderr=subprocess.STDOUT,check=True)
            print('FINISHED_TRAIN',arm,rep,read(directory/'best.json')['step'],flush=True)
    frozen={}
    for rep in (0,1):
        for arm in ('control','aux'):
            d=P/'training'/f'{arm}_r{rep}';step=read(d/'best.json')['step']
            frozen[f'{arm}_r{rep}']=dict(best_step=step,selected=str(d/f'step{step:03}.bin'),
                  selected_sha=digest(d/f'step{step:03}.bin'),last=str(d/'step020.bin'),last_sha=digest(d/'step020.bin'))
    if (P/'FINAL_SELECTION_FROZEN.json').exists():assert read(P/'FINAL_SELECTION_FROZEN.json')==frozen
    else:save(P/'FINAL_SELECTION_FROZEN.json',frozen)
    train_summary={key:read(P/'training'/key/'COMPLETE.json') for key in frozen}
    save(P/'G2_RESULTS.json',train_summary)
    (P/'G2_ACCEPTANCE_ZH.md').write_text('# G2 配对训练验收\n\n4个训练运行已完成，每个20批×224完整局，总17,920局。主体/初始化/种子/更新预算相同，仅辅助损失开关不同；两组后续轨迹自然不同，不能把两组强行喂同一批旧轨迹做PPO。\n\n模型选择已在 FINAL_SELECTION_FROZEN.json 冻结，之后才读取最终100种子的对战结果。G2只证明训练完整且对照公平，不证明提升。\n',encoding='utf8')
    configure_torch();pool=runtime.make_pool();base=P/'final';base.mkdir(exist_ok=True)
    if not (base/'keep/summary.json').exists():evaluate(pool,base/'keep')
    forecasts={}
    for key,choice in frozen.items():
        for label,checkpoint,mode in [('selected',choice['selected'],1),('last_greedy',choice['last'],1),('last_sample',choice['last'],2)]:
            directory=base/f'{key}_{label}'
            if not (directory/'summary.json').exists():evaluate(pool,directory,checkpoint,mode)
            if label=='last_sample':
                m=AuxModel().cuda();m.load_state_dict(torch.load(Path(checkpoint).with_suffix('.pt'),weights_only=True))
                with np.load(directory/'decisions.npz')as data:forecasts[key]=forecast_metrics(m,data)
    save(P/'FORECAST_RESULTS.json',forecasts)
    for f,h in protocol['hashes'].items():assert digest(f)==h,f
    save(P/'RUN_COMPLETE.json',dict(status='PASS',training_games=17920,final_games=18200,seconds=time.perf_counter()-tic,
         preserved_frozen_files=True,selection_before_final=True))
    subprocess.run([sys.executable,str(P/'report.py')],check=True)
    print('ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
