"""Read-only evidence audit after the predeclared training and final panel.

This script does not alter policies, choose checkpoints or start new rollouts.
"""
from run100_common import *
from bias2_model import Bias2Model
import torch


def check_games(path, start, count):
    rows=read(path/'games.json')
    expected={(s,seat,n)for s in range(start,start+count)for n in NAMES for seat in (0,1)}
    assert len(rows)==count*14
    assert {(r['seed'],r['seat'],r['opponent'])for r in rows}==expected
    for r in rows:
        assert not r['error'] and r['steps']==719
        assert r['reference_calls']==0 and r['plan_calls']==30 and r['execute_calls']==719
        assert r['win']==(r['margin']>0)
    return rows


def check_records(path, games):
    with np.load(path/'decisions.npz')as archive:
        data={k:archive[k]for k in ('choice','mask','game','day','reward','changed')}
    n=len(games);assert len(data['choice'])==n*30
    assert np.array_equal(data['day'],np.tile(np.arange(30),n))
    assert np.array_equal(data['game'],np.repeat(np.arange(n),30))
    assert np.all(data['mask'][np.arange(n*30),data['choice']]>0)
    assert np.isfinite(data['reward']).all()
    rewards=data['reward'].reshape(n,30)
    assert np.all(rewards[:,:-1]==0)
    margins=np.array([r['margin']for r in games])
    expected=np.sign(margins)+.1*np.tanh(margins/50000.)
    assert np.max(abs(rewards[:,-1]-expected))<1e-6
    s=read(path/'summary.json')
    assert s['non_keep']==int(np.sum(data['choice']!=0))
    assert s['changed']==int(np.sum(data['changed']))
    return dict(games=n,decisions=n*30,nonkeep=s['non_keep'],changed=s['changed'])


def checkpoint_parity(directory, round_number):
    # rollout N was sampled with checkpoint N-1, never checkpoint N.
    pt=directory/f'step{round_number-1:03}.pt'
    state=torch.load(pt,weights_only=False)
    assert state['step']==round_number-1
    assert state['bin_sha']==digest(pt.with_suffix('.bin'))
    m=Bias2Model().cuda();m.load_state_dict(state['model']);m.eval()
    with np.load(directory/f'rollout_{round_number:03}/decisions.npz')as archive:
        fields=('global','candidates','mask','choice','logp','value','probability')
        # A deterministic, evenly spaced check including both ends of the batch.
        ix=np.linspace(0,len(archive['choice'])-1,1024,dtype=int)
        d={k:archive[k][ix]for k in fields}
    with torch.no_grad():
        dist,v=m(torch.as_tensor(d['global'],device='cuda'),torch.as_tensor(d['candidates'],device='cuda'),torch.as_tensor(d['mask'],device='cuda'))
        logp=dist.log_prob(torch.as_tensor(d['choice'],device='cuda').long()).cpu().numpy()
    errors=dict(prob=float(np.max(abs(dist.probs.cpu().numpy()-d['probability']))),
                value=float(np.max(abs(v.cpu().numpy()-d['value']))),
                logp=float(np.max(abs(logp-d['logp']))),
                ratio=float(np.max(abs(np.exp(logp-d['logp'])-1))))
    assert max(errors.values())<2e-5,errors
    return errors


def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS'
    check_hashes();configure_torch()
    training={};total=0;rollout_parity={};paired_initial={}
    for rep in (0,1):
        a=P/'training'/f'control_r{rep}';b=P/'training'/f'aux_r{rep}'
        assert digest(a/'step000.bin')==digest(b/'step000.bin')==digest(OLD/'training'/f'f3_r{rep}/step000.bin')
        with np.load(a/'rollout_001/decisions.npz')as aa,np.load(b/'rollout_001/decisions.npz')as bb:
            assert aa.files==bb.files
            for k in aa.files:assert np.array_equal(aa[k],bb[k]),k
        ra=read(a/'rollout_001/games.json');rb=read(b/'rollout_001/games.json')
        for x,y in zip(ra,rb):
            for k in ('seed','seat','opponent','cash','margin','win','steps'):assert x[k]==y[k]
        paired_initial[str(rep)]=True
        for arm in ('control','aux'):
            name=f'{arm}_r{rep}';directory=P/'training'/name
            receipt=read(directory/'COMPLETE.json');assert receipt['status']=='PASS'
            assert receipt['training_games']==22400 and receipt['keep_bonus']==2
            history=receipt['history'];assert [r['step']for r in history]==list(range(1,101))
            assert len(list(directory.glob('rollout_*')))==100
            changed_hashes=0
            for step in range(1,101):
                folder=directory/f'rollout_{step:03}'
                previous=directory/f'step{step-1:03}.bin';current=directory/f'step{step:03}.bin'
                assert read(folder/'input.json')['sha']==digest(previous)
                assert previous.stat().st_size==70402*4 and current.stat().st_size==70402*4
                assert (directory/f'step{step:03}.pt').exists()
                changed_hashes+=digest(previous)!=digest(current)
                rows=check_games(folder,69600000+rep*100000+(step-1)*16,16)
                check_records(folder,rows);total+=len(rows)
            assert changed_hashes==100
            selection=read(directory/'selection.json')
            assert [r['step']for r in selection]==[0,20,40,60,80,100]
            assert max(selection,key=lambda r:tuple(r['score']))==receipt['best']
            for step in (0,20,40,60,80,100):
                rows=check_games(directory/f'dev_{step:03}',69800000,32)
                check_records(directory/f'dev_{step:03}',rows)
            rollout_parity[name]={str(n):checkpoint_parity(directory,n)for n in (1,20,100)}
            training[name]=dict(games=22400,rounds=100,updates_changed_weights=changed_hashes,best_step=receipt['best']['step'])
    assert total==89600
    train_seeds=set(range(69600000,69601600))|set(range(69700000,69701600))
    dev_seeds=set(range(69800000,69800032));final_seeds=set(range(69900000,69900100))
    assert not(train_seeds&dev_seeds or train_seeds&final_seeds or dev_seeds&final_seeds)
    frozen=P/'FINAL_SELECTION_FROZEN.json';chosen=read(frozen)
    final_dirs=sorted((P/'final').iterdir());assert len(final_dirs)==21
    finals={};behavior={}
    for directory in final_dirs:
        assert frozen.stat().st_mtime<=(directory/'games.json').stat().st_mtime
        rows=check_games(directory,69900000,100)
        finals[directory.name]=check_records(directory,rows)
        provenance=read(directory/'provenance.json')
        assert provenance['keep_bonus']==2 and provenance['library_sha']==digest(P/'build/keep2.so')
        if provenance['checkpoint']:assert provenance['sha256']==digest(provenance['checkpoint'])
        with np.load(directory/'decisions.npz')as archive:
            choice=archive['choice'];mask=archive['mask'];day=archive['day'];changed=archive['changed']
        behavior[directory.name]=dict(choice_counts=np.bincount(choice,minlength=10).tolist(),
              effective=int((mask.sum(1)>1).sum()),
              by_day=[dict(day=d,total=int((day==d).sum()),nonkeep=int(((day==d)&(choice!=0)).sum()),changed=int(changed[day==d].sum()))for d in range(30)])
    assert sum(r['games']for r in finals.values())==29400
    for name,config in chosen.items():
        assert config['best_step']==training[name]['best_step']
        for f,h in config['hashes'].items():assert digest(f)==h
        assert read(P/'final'/f'{name}_selected/provenance.json')['checkpoint']==config['selected']
    check_hashes()
    save(P/'FINAL_AUDIT.json',dict(status='PASS',training=training,training_games=total,final_games=29400,
         paired_initial_rollouts_identical=paired_initial,cpp_torch_onpolicy=rollout_parity,
         splits_disjoint=True,all_selection_before_final=True,finals=finals,old_files_unchanged=True))
    save(P/'BEHAVIOR_DIAGNOSTIC.json',behavior)
    print('FINAL_AUDIT_PASS',total,29400,flush=True)


if __name__=='__main__':main()
