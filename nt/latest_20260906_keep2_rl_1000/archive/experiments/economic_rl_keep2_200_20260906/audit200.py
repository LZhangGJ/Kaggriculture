"""Independent saved-evidence checks; no policy updates or new rollouts."""
from common200 import *
from audit100 import check_games,check_records,checkpoint_parity


def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS';check_hashes();configure_torch()
    preserved_count=check_source_manifest();new_games=0;training={};parity={}
    for rep in (0,1):
        for arm in ('control','aux'):
            name=f'{arm}_r{rep}';directory=P/'training'/name;old=SOURCE/'training'/name
            for ext in ('pt','bin'):assert digest(directory/f'step100.{ext}')==digest(old/f'step100.{ext}')
            initial=read(directory/'INITIAL_STATE.json')
            assert initial['sha']==digest(old/'step100.pt') and initial['optimizer']['steps']==[5600]
            receipt=read(directory/'COMPLETE.json')
            assert receipt['status']=='PASS' and receipt['new_training_games']==22400 and receipt['total_training_games']==44800
            assert receipt['final_optimizer']['steps']==[11200]
            assert [h['step']for h in receipt['history']]==list(range(101,201))
            assert len(list(directory.glob('rollout_*')))==100
            for step in range(101,201):
                previous=directory/f'step{step-1:03}.bin';current=directory/f'step{step:03}.bin'
                folder=directory/f'rollout_{step:03}'
                assert read(folder/'input.json')['sha']==digest(previous)
                assert current.stat().st_size==70402*4 and digest(previous)!=digest(current)
                state=torch.load(directory/f'step{step:03}.pt',weights_only=False,map_location='cpu')
                assert state['step']==step and state['bin_sha']==digest(current)
                assert {int(s['step'].item())for s in state['optimizer']['state'].values()}=={step*56}
                rows=check_games(folder,69600000+rep*100000+(step-1)*16,16)
                check_records(folder,rows);new_games+=len(rows)
            selection=read(directory/'selection.json')
            assert [r['step']for r in selection]==list(range(0,201,20))
            assert selection[:6]==read(old/'selection.json')
            assert max(selection,key=lambda s:tuple(s['score']))==receipt['best']
            for step in (120,140,160,180,200):
                rows=check_games(directory/f'dev_{step:03}',69800000,32)
                check_records(directory/f'dev_{step:03}',rows)
            parity[name]={str(step):checkpoint_parity(directory,step)for step in (101,150,200)}
            training[name]=dict(new_games=22400,total_games=44800,source_step=100,end_step=200,
                source_optimizer_updates=5600,final_optimizer_updates=11200,best_step=receipt['best']['step'])
    assert new_games==89600
    old_train=set(range(69600000,69601600))|set(range(69700000,69701600))
    new_train=set(range(69601600,69603200))|set(range(69701600,69703200))
    dev=set(range(69800000,69800032));old_final=set(range(69900000,69900100));new_final=set(range(70100000,70100100))
    groups=[old_train,new_train,dev,old_final,new_final]
    for i,a in enumerate(groups):
        for b in groups[i+1:]:assert not(a&b)
    frozen=P/'FINAL_SELECTION_FROZEN.json';chosen=read(frozen)
    dirs=sorted((P/'final').iterdir());assert len(dirs)==21;finals={};behavior={}
    for directory in dirs:
        assert frozen.stat().st_mtime<=(directory/'games.json').stat().st_mtime
        rows=check_games(directory,70100000,100);finals[directory.name]=check_records(directory,rows)
        prov=read(directory/'provenance.json')
        assert prov['seed_start']==70100000 and prov['seeds']==100 and prov['keep_bonus']==2
        assert prov['library_sha']==digest(SOURCE/'build/keep2.so')
        if prov['checkpoint']:assert prov['sha256']==digest(prov['checkpoint'])
        with np.load(directory/'decisions.npz')as z:
            choice=z['choice'];mask=z['mask'];day=z['day'];changed=z['changed']
        behavior[directory.name]=dict(choice_counts=np.bincount(choice,minlength=10).tolist(),effective=int((mask.sum(1)>1).sum()),
            by_day=[dict(day=d,total=int((day==d).sum()),nonkeep=int(((day==d)&(choice!=0)).sum()),changed=int(changed[day==d].sum()))for d in range(30)])
    assert sum(r['games']for r in finals.values())==29400
    for name,c in chosen.items():
        assert c['best_step']==training[name]['best_step']
        assert c['selected']==str(chosen_path(name,c['best_step']))
        for f,h in c['hashes'].items():assert digest(f)==h
        for label,key in [('selected','selected'),('step100_greedy','step100'),('step100_sample','step100'),('step200_greedy','step200'),('step200_sample','step200')]:
            prov=read(P/'final'/f'{name}_{label}/provenance.json')
            assert prov['checkpoint']==c[key] and prov['mode']==(2 if label.endswith('sample')else 1)
    check_hashes()
    save(P/'FINAL_AUDIT.json',dict(status='PASS',source_manifest_files_verified=preserved_count,training=training,new_training_games=new_games,
        total_training_games=179200,final_games=29400,cpp_torch_onpolicy=parity,optimizer_continued_exactly=True,
        train_dev_oldfinal_newfinal_disjoint=True,selection_before_final=True,old_files_unchanged=True,finals=finals))
    save(P/'BEHAVIOR_DIAGNOSTIC.json',behavior)
    print('AUDIT200_PASS',new_games,29400,flush=True)


if __name__=='__main__':main()
