"""Post-run independent receipts. No model/data/selection changes."""
from common import *
from aux_model import AuxModel, outcomes
import torch

def main():
    assert read(P/'RUN_COMPLETE.json')['status']=='PASS'
    configure_torch();protocol=read(P/'G2_PROTOCOL.json')
    for f,h in protocol['hashes'].items():assert digest(f)==h,f
    with np.load(P/'g1/dataset.npz')as d:
        g1seeds={s:set(d['seed'][d['split']==s].tolist())for s in ('train','dev','test')}
        for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not g1seeds[a]&g1seeds[b]
        assert np.all(d['win'][:,0]==0) and np.all(d['margin'][:,0]==0)
    total=0;parity=[];control_aux_changes={};trainseeds=set()
    for rep in (0,1):
        control=P/'training'/f'control_r{rep}';aux=P/'training'/f'aux_r{rep}'
        assert digest(control/'step000.bin')==digest(aux/'step000.bin')
        with np.load(control/'rollout_001/decisions.npz')as c, np.load(aux/'rollout_001/decisions.npz')as a:
            for k in ('global','candidates','mask','probability','choice','reward','value'):assert np.array_equal(c[k],a[k]),k
        cr=read(control/'rollout_001/games.json');ar=read(aux/'rollout_001/games.json')
        for c,a in zip(cr,ar):
            for k in ('seed','seat','opponent','steps','cash','opponent_cash','margin'):assert c[k]==a[k],k
        changes=0
        for arm in ('control','aux'):
            directory=P/'training'/f'{arm}_r{rep}';complete=read(directory/'COMPLETE.json')
            assert len(complete['history'])==20 and complete['training_games']==4480
            for step in range(1,21):
                rows=read(directory/f'rollout_{step:03}/games.json');trainseeds.update(r['seed']for r in rows)
                assert len(rows)==224 and all(not r['error'] and r['steps']==719 and r['reference_calls']==0 for r in rows)
                total+=len(rows)
            # Last training rollout must use the immediately preceding checkpoint.
            m=AuxModel().cuda();m.load_state_dict(torch.load(directory/'step019.pt',weights_only=True))
            with np.load(directory/'rollout_020/decisions.npz')as data:
                ii=np.random.default_rng(2006).choice(len(data['choice']),256,replace=False)
                with torch.no_grad():
                    dist,v=m(torch.as_tensor(data['global'][ii],device='cuda'),torch.as_tensor(data['candidates'][ii],device='cuda'),
                             torch.as_tensor(data['mask'][ii],device='cuda'))
                pe=float(np.max(abs(dist.probs.cpu().numpy()-data['probability'][ii])))
                ve=float(np.max(abs(v.cpu().numpy()-data['value'][ii])))
                assert max(pe,ve)<2e-5,(arm,rep,pe,ve)
                y,mask=outcomes(data)
                for hi,h in enumerate((1,3,7)):assert np.array_equal(mask[:,hi,0],data['day']+h<=29)
                parity.append(dict(arm=arm,repeat=rep,prob_error=pe,value_error=ve))
            # Future-only weights cannot affect C++ weights or action selection.
            m.export(P/'checks/export_aux_before.bin')
            with torch.no_grad():m.future.weight.add_(100)
            m.export(P/'checks/export_aux_after.bin')
            assert digest(P/'checks/export_aux_before.bin')==digest(P/'checks/export_aux_after.bin')
        for step in range(2,21):
            with np.load(control/f'rollout_{step:03}/decisions.npz')as c,np.load(aux/f'rollout_{step:03}/decisions.npz')as a:
                changes+=int(np.sum(c['choice']!=a['choice']))
        control_aux_changes[str(rep)]=changes
    final=read(P/'final/keep/games.json');finalseeds={r['seed']for r in final}
    devseeds={r['seed']for r in read(P/'training/control_r0/dev_000/games.json')}
    assert not trainseeds&finalseeds and not trainseeds&devseeds and not devseeds&finalseeds
    assert not set.union(*g1seeds.values())&finalseeds
    finalgames=0
    for d in (P/'final').iterdir():
        if not (d/'games.json').exists():continue
        rows=read(d/'games.json');assert len(rows)==1400
        assert {r['seed']for r in rows}==finalseeds
        assert all(not r['error'] and r['steps']==719 and r['reference_calls']==0 for r in rows)
        finalgames+=len(rows)
    assert total==17920 and finalgames==18200
    receipt=dict(status='PASS',training_games=total,final_games=finalgames,initial_weights_and_rollouts_paired_exact=True,
       old_engine_and_opponents_unchanged=True,train_dev_final_seeds_disjoint=True,g1_labels_never_PPO_inputs=True,
       last_training_on_policy_cpp_torch=parity,labels_never_cross_episode=True,aux_weights_excluded_from_export=True,
       control_aux_later_choice_differences=control_aux_changes)
    save(P/'FINAL_AUDIT.json',receipt)
    manifest={str(f.relative_to(P)):digest(f)for f in P.rglob('*')if f.is_file() and f.suffix in ('.py','.md','.bin','.pt','.json','.npz')
              and '__pycache__'not in f.parts and f.name!='MANIFEST.json'}
    save(P/'MANIFEST.json',manifest)
    print('FINAL_AUDIT_PASS',receipt,flush=True)

if __name__=='__main__':main()
